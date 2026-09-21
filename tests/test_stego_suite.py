"""
Comprehensive Unit and Integration Test Suite for Stego Diffusion Pipeline
Tests exact fixed-point inversion, distribution-preserving codec, rate-adaptive ECC, and pipeline end-to-end.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from codec.distribution_preserving import DistributionPreservingCodec
from codec.ecc_engine import ECCEngine
from inversion.fixed_point_ddim import FixedPointDDIMInverter
from pipelines.stego_diffusion_pipeline import (
    StegoDiffusionPipeline,
    compute_psnr,
    compute_ssim,
)


class MockDenoiser(nn.Module):
    """Deterministic lightweight model for fast reproducible unit testing."""
    def __init__(self, channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight.normal_(0, 0.005)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale = (t.float().view(-1, 1, 1, 1) + 1.0) / 1000.0
        return self.conv(x) * 0.05 + x * scale * 0.005


class TestExactFixedPointDDIM(unittest.TestCase):
    def setUp(self):
        self.model = MockDenoiser()
        self.betas = torch.linspace(1e-4, 0.02, 1000)
        self.inverter = FixedPointDDIMInverter(tol=1e-6, max_iter=5)

    def test_bidirectional_consistency(self):
        x0 = torch.randn(1, 3, 16, 16)
        seq = [0, 10, 20, 30, 40, 50]
        res = self.inverter.verify_consistency(x0, seq, self.model, self.betas)
        self.assertTrue(res["is_consistent"])
        self.assertLess(res["l2_error"], 1e-4)
        self.assertGreater(res["psnr"], 40.0)

    def test_fp32_upcasting(self):
        x0 = torch.randn(1, 3, 16, 16, dtype=torch.float32)
        seq = [0, 20, 40]
        xs, _ = self.inverter.forward(x0, seq, self.model, self.betas)
        for s in xs:
            self.assertEqual(s.dtype, torch.float32)


class TestRateAdaptiveECCEngine(unittest.TestCase):
    def setUp(self):
        self.rates = ["1/2", "2/3", "3/4", "5/6"]

    def test_clean_encode_decode_all_rates(self):
        msg = b"Confidential Payload 2026"
        for r in self.rates:
            ecc = ECCEngine(default_rate=r)
            coded = ecc.encode(msg, rate=r)
            recovered, valid = ecc.decode(coded, rate=r)
            self.assertTrue(valid, f"Rate {r} validation failed.")
            self.assertEqual(recovered, msg, f"Rate {r} payload mismatch.")

    def test_error_correction_tolerance(self):
        msg = b"Zero Bit Error Rate Target"
        ecc = ECCEngine(default_rate="1/2")
        coded = ecc.encode(msg, rate="1/2")

        # Inject simulated bit error (burst of 2 adjacent bits)
        noisy = coded.copy()
        noisy[10] ^= 1
        noisy[11] ^= 1

        recovered, valid = ecc.decode(noisy, rate="1/2")
        self.assertTrue(valid)
        self.assertEqual(recovered, msg)

    def test_interleaver_deinterleaver(self):
        ecc = ECCEngine(interleaver_seed=42)
        bits = np.random.randint(0, 2, size=100, dtype=np.uint8)
        interleaved = ecc.interleave(bits)
        deinterleaved = ecc.deinterleave(interleaved)
        np.testing.assert_array_equal(bits, deinterleaved)


class TestDistributionPreservingCodec(unittest.TestCase):
    def setUp(self):
        self.codec = DistributionPreservingCodec(default_seed=42)

    def test_exact_reversibility(self):
        shape = (1, 3, 32, 32)
        bits = np.random.randint(0, 2, size=500, dtype=np.uint8)
        latents = self.codec.encode_bits_to_latents(bits, shape, seed=42)
        extracted = self.codec.decode_latents_to_bits(latents, len(bits), seed=42)
        np.testing.assert_array_equal(bits, extracted)

    def test_statistical_indistinguishability(self):
        shape = (1, 3, 64, 64)
        bits = np.random.randint(0, 2, size=1000, dtype=np.uint8)
        ad_stats = []
        ks_pvals = []
        for s in [42, 100, 2026]:
            latents = self.codec.encode_bits_to_latents(bits, shape, seed=s)
            res = self.codec.test_statistical_indistinguishability(latents)
            ad_stats.append(res["ad_statistic"])
            ks_pvals.append(res["ks_pvalue"])
        
        mean_ad = float(np.mean(ad_stats))
        mean_ks = float(np.mean(ks_pvals))
        self.assertLess(mean_ad, 0.752, f"Mean Anderson-Darling stat too high: {mean_ad}")
        self.assertGreater(mean_ks, 0.05, f"Mean KS p-value too low: {mean_ks}")


class TestStegoDiffusionPipeline(unittest.TestCase):
    def setUp(self):
        self.model = MockDenoiser()
        self.betas = torch.linspace(1e-4, 0.02, 1000)
        self.pipeline = StegoDiffusionPipeline(
            model=self.model,
            betas=self.betas,
            image_shape=(3, 32, 32),
        )

    def test_end_to_end_pipeline(self):
        secret = b"Pipeline-End-to-End-Verification"
        stego_img, metrics = self.pipeline.embed(
            secret_message=secret,
            steps=10,
            ecc_rate="1/2",
            seed=42,
        )
        self.assertEqual(metrics.ber, 0.0)
        self.assertTrue(metrics.extraction_success)
        self.assertGreaterEqual(metrics.psnr, 42.0)
        self.assertGreaterEqual(metrics.ssim, 0.98)

        recovered, valid = self.pipeline.extract(
            stego_img,
            steps=10,
            expected_bit_length=metrics.payload_size_bits,
            seed=42,
        )
        self.assertTrue(valid)
        self.assertEqual(recovered, secret)


if __name__ == "__main__":
    unittest.main()
