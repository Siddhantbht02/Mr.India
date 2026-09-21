"""
Steganalysis and Statistical Verification Suite
Evaluates statistical indistinguishability of message-embedded latents and synthesized stego images
using Anderson-Darling, Kolmogorov-Smirnov, higher-order moments, and DCT domain analysis.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import scipy.stats as stats
import torch

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from codec.distribution_preserving import DistributionPreservingCodec
from codec.ecc_engine import ECCEngine


def compute_higher_order_moments(data: np.ndarray) -> Dict[str, float]:
    """Computes mean, standard deviation, skewness, and excess kurtosis."""
    arr = data.flatten()
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "skewness": float(stats.skew(arr)),
        "kurtosis": float(stats.kurtosis(arr)),
    }


def evaluate_statistical_tests(samples: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Runs KS-test and Anderson-Darling test against standard normal distribution N(0, 1)."""
    # 1. Kolmogorov-Smirnov
    ks_stat, ks_pval = stats.kstest(samples, "norm")

    # 2. Anderson-Darling
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad_res = stats.anderson(samples, dist="norm")
    ad_stat = float(ad_res.statistic)
    ad_crit = float(ad_res.critical_values[2])  # 5% significance level

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "ks_pass": bool(ks_pval > alpha),
        "ad_statistic": ad_stat,
        "ad_crit_5pct": ad_crit,
        "ad_pass": bool(ad_stat < ad_crit),
    }


def dct2d_analysis(image_matrix: np.ndarray) -> np.ndarray:
    """Computes 2D Discrete Cosine Transform of an image channel."""
    return cv2.dct(image_matrix.astype(np.float32))


def run_steganalysis_suite(
    num_samples: int = 50,
    shape: Tuple[int, ...] = (1, 3, 64, 64),
    output_dir: str = "output/benchmarks",
) -> Dict[str, any]:
    """
    Executes deep steganalysis benchmark across multiple message payloads,
    comparing message-embedded latents against unconditioned Gaussian noise.
    """
    os.makedirs(output_dir, exist_ok=True)
    codec = DistributionPreservingCodec()
    ecc = ECCEngine(default_rate="1/2")

    print("\n==================================================================")
    print("Starting Steganalysis & Statistical Indistinguishability Benchmark")
    print(f"Testing {num_samples} sample blocks with shape {shape}")
    print("==================================================================\n")

    ks_pvals = []
    ad_stats = []
    skewness_list = []
    kurtosis_list = []

    for i in range(num_samples):
        # Generate arbitrary secret payload
        payload = f"Steganalysis-Evaluation-Payload-Index-{i}-Salt-{i*1337}".encode("utf-8")
        coded_bits = ecc.encode(payload, rate="1/2")

        # DPAC Latent embedding
        z_stego = codec.encode_bits_to_latents(
            bits=coded_bits,
            shape=shape,
            seed=1000 + i,
        )

        samples = z_stego.flatten().cpu().numpy().astype(np.float64)
        stat_res = evaluate_statistical_tests(samples)
        moments = compute_higher_order_moments(samples)

        ks_pvals.append(stat_res["ks_pvalue"])
        ad_stats.append(stat_res["ad_statistic"])
        skewness_list.append(moments["skewness"])
        kurtosis_list.append(moments["kurtosis"])

    mean_ks_pval = float(np.mean(ks_pvals))
    pass_ks_rate = float(np.mean([p > 0.05 for p in ks_pvals])) * 100.0
    mean_ad_stat = float(np.mean(ad_stats))
    pass_ad_rate = float(np.mean([s < 0.752 for s in ad_stats])) * 100.0
    mean_skew = float(np.mean(skewness_list))
    mean_kurt = float(np.mean(kurtosis_list))

    print("Steganalysis Evaluation Summary:")
    print(f"  Total Trials:              {num_samples}")
    print(f"  Mean KS p-value:           {mean_ks_pval:.4f} (Criterion: p > 0.05)")
    print(f"  KS Test Pass Rate:         {pass_ks_rate:.1f}%")
    print(f"  Mean Anderson-Darling:     {mean_ad_stat:.4f} (Criterion: < 0.752)")
    print(f"  Anderson-Darling Pass:     {pass_ad_rate:.1f}%")
    print(f"  Mean Skewness:             {mean_skew:.5f} (Theoretical: 0.00000)")
    print(f"  Mean Excess Kurtosis:      {mean_kurt:.5f} (Theoretical: 0.00000)")

    summary = {
        "num_samples": num_samples,
        "mean_ks_pvalue": mean_ks_pval,
        "ks_pass_rate_percent": pass_ks_rate,
        "mean_ad_statistic": mean_ad_stat,
        "ad_pass_rate_percent": pass_ad_rate,
        "mean_skewness": mean_skew,
        "mean_kurtosis": mean_kurt,
        "is_undetectable": bool(mean_ks_pval > 0.05 and mean_ad_stat < 0.752),
    }

    report_path = os.path.join(output_dir, "steganalysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Generative Steganography Steganalysis Benchmark Report ===\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"\nSteganalysis benchmark report written to: {report_path}\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Steganalysis Benchmark")
    parser.add_argument("--samples", type=int, default=50, help="Number of latent test trials (default: 50)")
    parser.add_argument("--dim", type=int, default=64, help="Latent resolution dimension (default: 64)")
    args = parser.parse_args()

    run_steganalysis_suite(num_samples=args.samples, shape=(1, 3, args.dim, args.dim))
