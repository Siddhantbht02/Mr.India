"""
Distribution-Preserving Latent Bit Embedder (DPAC)
Guarantees message-embedded diffusion latents follow an exact isotropic Gaussian N(0, I)
distribution using recursive inverse CDF transformation, whitening, and range quantization.
"""

from typing import Any, Dict, Optional, Tuple, Union
import warnings
import numpy as np
import scipy.special as special
import scipy.stats as stats
import torch


class DistributionPreservingCodec:
    """
    Distribution-Preserving Arithmetic Coding (DPAC) Embedder & Extractor.
    
    Transforms secret bits into samples statistically indistinguishable from N(0, I).
    Applies synchronized pseudo-random keystream whitening and inverse CDF mapping.
    """

    def __init__(self, eps: float = 1e-9, default_seed: int = 42):
        self.eps = eps
        self.default_seed = default_seed

    @staticmethod
    def _standard_normal_cdf(x: np.ndarray) -> np.ndarray:
        """Computes Phi(x) using high-precision error function."""
        return 0.5 * (1.0 + special.erf(x / np.sqrt(2.0)))

    @staticmethod
    def _standard_normal_icdf(u: np.ndarray) -> np.ndarray:
        """Computes Phi^(-1)(u) using high-precision quantile function ndtri."""
        u_clamped = np.clip(u, 1e-12, 1.0 - 1e-12)
        return special.ndtri(u_clamped)

    def _get_whitening_keystream(self, length: int, seed: int) -> np.ndarray:
        """Generates synchronized binary pseudo-random sequence for payload whitening."""
        rng = np.random.RandomState(seed ^ 0x5A5A5A5A)
        return rng.randint(0, 2, size=length, dtype=np.uint8)

    def encode_bits_to_latents(
        self,
        bits: Union[np.ndarray, torch.Tensor],
        shape: Tuple[int, ...],
        seed: Optional[int] = None,
        bits_per_element: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Embeds binary secret bits into Gaussian latents matching shape (B, C, H, W).
        
        Args:
            bits: 1D array of binary ints {0, 1}.
            shape: Target latent tensor shape.
            seed: RNG seed for reproducible synchronized dithering and whitening.
            bits_per_element: Number of bits per coordinate (default: 1).
            device: PyTorch device.
            
        Returns:
            torch.Tensor of shape `shape` distributed as isotropic Gaussian N(0, I).
        """
        dev = device if device is not None else torch.device("cpu")
        rng_seed = seed if seed is not None else self.default_seed
        rng = np.random.RandomState(rng_seed)

        if isinstance(bits, torch.Tensor):
            bits_np = bits.detach().cpu().numpy().astype(np.uint8).flatten()
        else:
            bits_np = np.asarray(bits, dtype=np.uint8).flatten()

        total_elements = int(np.prod(shape))
        num_symbols = 2 ** bits_per_element
        max_symbols = total_elements

        # Group bits into symbols of `bits_per_element`
        num_bits = len(bits_np)
        symbols_needed = (num_bits + bits_per_element - 1) // bits_per_element

        if symbols_needed > max_symbols:
            raise ValueError(
                f"Payload size {num_bits} bits exceeds latent capacity {max_symbols * bits_per_element} bits."
            )

        # Pad bits to multiple of bits_per_element
        pad_len = symbols_needed * bits_per_element - num_bits
        if pad_len > 0:
            bits_np = np.pad(bits_np, (0, pad_len), mode="constant", constant_values=0)

        # Apply synchronized whitening to guarantee 50/50 bit balance regardless of payload content
        keystream = self._get_whitening_keystream(len(bits_np), rng_seed)
        whitened_bits = bits_np ^ keystream

        # Convert whitened bits to symbol integers in [0, num_symbols - 1]
        reshaped_bits = whitened_bits.reshape(-1, bits_per_element)
        powers = 2 ** np.arange(bits_per_element - 1, -1, -1)
        payload_symbols = np.dot(reshaped_bits, powers).astype(np.int64)

        # Synchronized pseudo-random uniform dithering
        v = rng.uniform(self.eps, 1.0 - self.eps, size=total_elements).astype(np.float64)

        # Embed symbols into uniform intervals [symbol / num_symbols, (symbol + 1) / num_symbols]
        u = np.zeros(total_elements, dtype=np.float64)
        interval_width = 1.0 / num_symbols
        
        # Message carrying portion
        for idx, sym in enumerate(payload_symbols):
            u[idx] = sym * interval_width + v[idx] * interval_width

        # Non-payload carrying portion follows standard uniform noise
        if symbols_needed < total_elements:
            u[symbols_needed:] = v[symbols_needed:]

        # Transform uniform variables through standard normal inverse CDF
        z_np = self._standard_normal_icdf(u).reshape(shape)
        z = torch.from_numpy(z_np).to(device=dev, dtype=torch.float32)

        return z

    def decode_latents_to_bits(
        self,
        latents: torch.Tensor,
        bit_length: int,
        seed: Optional[int] = None,
        bits_per_element: int = 1,
    ) -> np.ndarray:
        """
        Extracts secret bits from latent tensor.
        
        Args:
            latents: Tensor of inverted latents at T.
            bit_length: Total number of bits to extract.
            seed: Shared RNG seed.
            bits_per_element: Number of bits per coordinate.
            
        Returns:
            np.ndarray of shape (bit_length,) containing extracted bits {0, 1}.
        """
        rng_seed = seed if seed is not None else self.default_seed
        z_flat = latents.detach().to(torch.float32).cpu().numpy().flatten()
        num_symbols = 2 ** bits_per_element
        symbols_needed = (bit_length + bits_per_element - 1) // bits_per_element

        if symbols_needed > len(z_flat):
            raise ValueError(
                f"Requested bit length {bit_length} exceeds available latent coordinates {len(z_flat)}."
            )

        z_sub = z_flat[:symbols_needed]
        
        # Transform through standard normal CDF back to [0, 1]
        u = self._standard_normal_cdf(z_sub)

        # Range quantize to recover symbols
        interval_width = 1.0 / num_symbols
        recovered_symbols = np.clip(np.floor(u / interval_width).astype(np.int64), 0, num_symbols - 1)

        # Convert symbols back to bit array
        extracted_bits = []
        for sym in recovered_symbols:
            sym_bits = [(sym >> shift) & 1 for shift in range(bits_per_element - 1, -1, -1)]
            extracted_bits.extend(sym_bits)

        extracted_bits = np.array(extracted_bits[:symbols_needed * bits_per_element], dtype=np.uint8)

        # Unwhiten with synchronized keystream
        keystream = self._get_whitening_keystream(len(extracted_bits), rng_seed)
        unwhitened_bits = extracted_bits ^ keystream

        return unwhitened_bits[:bit_length]

    def test_statistical_indistinguishability(
        self,
        latents: torch.Tensor,
        alpha_level: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Performs Kolmogorov-Smirnov and Anderson-Darling tests against standard normal distribution.
        
        Returns:
            Dictionary with test statistics, p-values, and pass/fail indicators.
        """
        samples = latents.detach().cpu().flatten().to(torch.float64).numpy()
        
        # Sample subset if too large for computational efficiency
        if len(samples) > 20000:
            eval_samples = np.random.choice(samples, size=20000, replace=False)
        else:
            eval_samples = samples

        # 1. Kolmogorov-Smirnov Test
        ks_stat, ks_pvalue = stats.kstest(eval_samples, "norm")

        # 2. Anderson-Darling Test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ad_result = stats.anderson(eval_samples, dist="norm")
        
        ad_stat = float(ad_result.statistic)
        crit_val_5pct = float(ad_result.critical_values[2])
        ad_pass = bool(ad_stat < crit_val_5pct)

        # 3. Higher-order moments
        mean_val = float(np.mean(samples))
        std_val = float(np.std(samples))
        skew_val = float(stats.skew(samples))
        kurt_val = float(stats.kurtosis(samples))

        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "ks_pass": bool(ks_pvalue > alpha_level),
            "ad_statistic": ad_stat,
            "ad_crit_value_5pct": crit_val_5pct,
            "ad_pass": ad_pass,
            "mean": mean_val,
            "std": std_val,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "is_statistically_indistinguishable": bool(ks_pvalue > alpha_level and ad_pass),
        }
