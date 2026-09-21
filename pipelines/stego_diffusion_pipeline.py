"""
Unified Stego Diffusion Pipeline
Orchestrator connecting ECC engine, distribution-preserving codec, and fixed-point DDIM inversion.
Supports end-to-end secret message embedding and extraction with comprehensive metric logging.
"""

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from codec.distribution_preserving import DistributionPreservingCodec
from codec.ecc_engine import ECCEngine
from inversion.base_inversion import compute_alpha
from inversion.fixed_point_ddim import FixedPointDDIMInverter


@dataclass
class StegoMetrics:
    """Dataclass storing fidelity, extraction accuracy, and latent reconstruction metrics."""
    ber: float  # Final payload Bit Error Rate after ECC (0.0 to 1.0)
    raw_ber: float  # Raw channel bit error rate before ECC
    extraction_success: bool  # True if CRC32 and SHA-256 integrity passed
    psnr: float  # PSNR against counterpart (dB)
    ssim: float  # SSIM against counterpart (0.0 to 1.0)
    lpips: float  # LPIPS perceptual distance
    latent_l2_error: float  # ||x_0 - x_hat_0||_2
    payload_size_bytes: int
    payload_size_bits: int
    inference_time_sec: float
    steps: int
    ecc_rate: str


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB for tensors in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * np.log10((max_val ** 2) / mse)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Computes Structural Similarity Index (SSIM) using PyTorch operations."""
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)

    channels = img1.size(1)
    
    # 1D Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # 2D Gaussian kernel
    kernel_2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    kernel = kernel_2d.repeat(channels, 1, 1, 1).to(device=img1.device, dtype=img1.dtype)

    pad = window_size // 2
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def compute_lpips_fallback(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Perceptual distance approximation: tries importing official lpips library,
    otherwise uses gradient-weighted high-frequency feature perceptual distance.
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex", verbose=False)
        i1 = (img1 * 2.0 - 1.0).clamp(-1.0, 1.0)
        i2 = (img2 * 2.0 - 1.0).clamp(-1.0, 1.0)
        with torch.no_grad():
            dist = loss_fn(i1, i2).item()
        return float(dist)
    except Exception:
        # High-frequency Sobel feature perceptual fallback
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img1.device).view(1, 1, 3, 3)
        c = img1.size(1)
        sx = sobel_x.repeat(c, 1, 1, 1)
        sy = sobel_y.repeat(c, 1, 1, 1)
        
        diff = img1 - img2
        grad_x = F.conv2d(diff, sx, padding=1, groups=c)
        grad_y = F.conv2d(diff, sy, padding=1, groups=c)
        dist = (diff.abs().mean() + (grad_x.abs().mean() + grad_y.abs().mean()) * 0.5).item()
        return float(dist)


class StegoDiffusionPipeline:
    """
    Unified Stego Diffusion Pipeline for high-capacity, zero-BER generative steganography.
    """

    def __init__(
        self,
        model: nn.Module,
        betas: torch.Tensor,
        inverter: Optional[FixedPointDDIMInverter] = None,
        codec: Optional[DistributionPreservingCodec] = None,
        ecc: Optional[ECCEngine] = None,
        image_shape: Tuple[int, int, int] = (3, 64, 64),
        device: Optional[torch.device] = None,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.betas = betas.to(self.device)
        self.inverter = inverter if inverter is not None else FixedPointDDIMInverter(device=self.device)
        self.codec = codec if codec is not None else DistributionPreservingCodec()
        self.ecc = ecc if ecc is not None else ECCEngine(default_rate="1/2")
        self.image_shape = image_shape
        self.num_timesteps = len(betas)

    def _get_timestep_sequence(self, steps: int) -> List[int]:
        """Subsamples total diffusion timesteps into a uniform progression."""
        step_stride = max(1, self.num_timesteps // steps)
        seq = [i * step_stride for i in range(steps)]
        return seq

    def embed(
        self,
        cover_image: Optional[Union[Image.Image, torch.Tensor, str]] = None,
        secret_message: bytes = b"",
        steps: int = 50,
        ecc_rate: str = "1/2",
        seed: int = 42,
    ) -> Tuple[Image.Image, StegoMetrics]:
        """
        Embeds secret message into a synthesized stego image.
        """
        start_time = time.time()
        c, h, w = self.image_shape
        latent_shape = (1, c, h, w)

        # 1. Error Correction & Header Framing
        coded_bits = self.ecc.encode(secret_message, rate=ecc_rate)
        payload_bits_len = len(coded_bits)

        # 2. Distribution-Preserving Latent Bit Embedding (x_T)
        z_stego = self.codec.encode_bits_to_latents(
            bits=coded_bits,
            shape=latent_shape,
            seed=seed,
            device=self.device,
        )

        seq = self._get_timestep_sequence(steps)

        # 3. Diffusion Generation (x_T -> x_0)
        with torch.no_grad():
            xs_stego, _ = self.inverter.backward(z_stego, seq, self.model, self.betas)

        x0_stego_raw = xs_stego[-1]

        # Normalize generated latents to image range [0, 1]
        stego_tensor = torch.clamp((x0_stego_raw + 1.0) / 2.0, 0.0, 1.0)

        # 4. Immediate Re-Extraction Verification across discrete image channel
        # Quantize to 8-bit RGB image
        stego_np = (stego_tensor[0].cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        stego_pil = Image.fromarray(stego_np)

        # Reconstructed image in [-1, 1]
        x0_channel = (torch.from_numpy(stego_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0).to(self.device)
        
        # Invert stego image back to latent space to test extraction accuracy
        xs_inverted, _ = self.inverter.forward(x0_channel, seq, self.model, self.betas)
        z_inverted = xs_inverted[-1]

        # Extract bits from inverted latent
        extracted_coded_bits = self.codec.decode_latents_to_bits(
            latents=z_inverted,
            bit_length=payload_bits_len,
            seed=seed,
        )

        # Calculate Raw Channel Bit Error Rate (BER)
        raw_bit_errors = int(np.sum(coded_bits != extracted_coded_bits))
        raw_ber = float(raw_bit_errors) / float(payload_bits_len)

        # ECC Decode
        recovered_msg, is_valid = self.ecc.decode(extracted_coded_bits, rate=ecc_rate)

        # Final Payload BER after error correction
        if is_valid and (recovered_msg == secret_message):
            payload_ber = 0.0
        elif recovered_msg:
            orig_bits = self.ecc._bytes_to_bits(secret_message)
            rec_bits = self.ecc._bytes_to_bits(recovered_msg)
            min_len = min(len(orig_bits), len(rec_bits))
            errs = np.sum(orig_bits[:min_len] != rec_bits[:min_len]) + abs(len(orig_bits) - len(rec_bits))
            payload_ber = float(errs) / float(max(len(orig_bits), 1))
        else:
            payload_ber = 1.0

        # 5. Measure Fidelity against Counterpart
        # Invert-regenerate cycle counterpart (evaluates exact reversibility fidelity)
        with torch.no_grad():
            xs_recon, _ = self.inverter.backward(z_inverted, seq, self.model, self.betas)
            recon_tensor = torch.clamp((xs_recon[-1] + 1.0) / 2.0, 0.0, 1.0)

        if cover_image is not None:
            # If explicit cover provided, compare against it
            if isinstance(cover_image, str):
                cover_image = Image.open(cover_image).convert("RGB")
            if isinstance(cover_image, Image.Image):
                c_np = np.array(cover_image.resize((w, h))).astype(np.float32) / 255.0
                counterpart_tensor = torch.from_numpy(c_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            elif isinstance(cover_image, torch.Tensor):
                counterpart_tensor = cover_image.to(self.device)
            else:
                counterpart_tensor = recon_tensor
        else:
            counterpart_tensor = recon_tensor

        psnr_val = compute_psnr(stego_tensor, counterpart_tensor)
        ssim_val = compute_ssim(stego_tensor, counterpart_tensor)
        lpips_val = compute_lpips_fallback(stego_tensor, counterpart_tensor)
        latent_l2 = torch.norm(x0_stego_raw - x0_channel, p=2).item() / (x0_stego_raw.numel() ** 0.5)

        elapsed = time.time() - start_time

        metrics = StegoMetrics(
            ber=payload_ber,
            raw_ber=raw_ber,
            extraction_success=is_valid and (recovered_msg == secret_message),
            psnr=psnr_val,
            ssim=ssim_val,
            lpips=lpips_val,
            latent_l2_error=latent_l2,
            payload_size_bytes=len(secret_message),
            payload_size_bits=payload_bits_len,
            inference_time_sec=elapsed,
            steps=steps,
            ecc_rate=ecc_rate,
        )

        return stego_pil, metrics

    def extract(
        self,
        stego_image: Union[Image.Image, torch.Tensor, str],
        steps: int = 50,
        expected_bit_length: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[bytes, bool]:
        """
        Extracts secret message from a stego image.
        """
        if isinstance(stego_image, str):
            stego_image = Image.open(stego_image).convert("RGB")

        if isinstance(stego_image, Image.Image):
            arr = np.array(stego_image).astype(np.float32)
            tensor = (torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0).to(self.device)
        elif isinstance(stego_image, torch.Tensor):
            tensor = stego_image.to(self.device)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.min() >= 0.0 and tensor.max() <= 1.0:
                tensor = tensor * 2.0 - 1.0
        else:
            raise TypeError("Unsupported image type.")

        seq = self._get_timestep_sequence(steps)

        # Invert from image x_0 to noise x_T
        with torch.no_grad():
            xs, _ = self.inverter.forward(tensor, seq, self.model, self.betas)
            z_inverted = xs[-1]

        # Extract bits
        total_capacity = int(np.prod(self.image_shape))
        bit_len = expected_bit_length if expected_bit_length is not None else total_capacity
        extracted_bits = self.codec.decode_latents_to_bits(
            latents=z_inverted,
            bit_length=bit_len,
            seed=seed,
        )

        # ECC Decode
        recovered_payload, is_valid = self.ecc.decode(extracted_bits)
        return recovered_payload, is_valid
