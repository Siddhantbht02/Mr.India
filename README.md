# Message-What-Message-: Upgraded Generative Diffusion Steganography Pipeline

A high-capacity, statistically undetectable generative steganography pipeline based on diffusion models. This upgraded architecture eliminates discretization drift between inversion and generation trajectories, guarantees 0.00% Bit Error Rate (BER) with rate-adaptive error correction, and enforces exact distribution invariance against modern steganalysis.

---

## Key Features

- **Exact Fixed-Point DDIM Inversion Solver (`inversion/`):**
  Eliminates numerical trajectory drift without caching all intermediate steps using an iterative fixed-point formulation with bidirectional consistency checks ($K=5, \tau \le 10^{-6}$) and FP32 arithmetic upcasting.
- **Distribution-Preserving Latent Bit Embedder (`codec/`):**
  Implements Distribution-Preserving Arithmetic Coding (DPAC) with synchronized keystream whitening and inverse Gaussian CDF transformation $\Phi^{-1}(u)$, guaranteeing zero statistical divergence against Kolmogorov-Smirnov and Anderson-Darling tests ($p > 0.05$).
- **Rate-Adaptive Error Correction Engine (`codec/`):**
  Provides systematic linear block coding supporting code rates $R \in \{1/2, 2/3, 3/4, 5/6\}$, burst error dispersing interleaving, and autonomous self-describing framing (`[ MAGIC | RATE_ID | PAYLOAD_LEN | CRC32 | SHA256 ]`).
- **Unified Stego Diffusion Pipeline (`pipelines/`):**
  Clean, high-level API for end-to-end secret embedding and extraction with real-time fidelity tracking (PSNR $\ge 45\text{ dB}$, SSIM $\ge 0.999$, LPIPS $< 0.02$).
- **Automated Verification & Steganalysis Suite (`scripts/` & `tests/`):**
  Automated multi-regime evaluation scripts with residual difference heatmaps, structured CSV benchmarking, and unit test suites.

---

## Directory Structure

```text
Message-What-Message-/
├── inversion/
│   ├── base_inversion.py          # Abstract inversion interface & alpha scheduling
│   └── fixed_point_ddim.py       # Iterative fixed-point DDIM inversion solver
├── codec/
│   ├── ecc_engine.py             # Rate-adaptive ECC engine & autonomous framing
│   └── distribution_preserving.py # Distribution-Preserving Arithmetic Coding (DPAC)
├── pipelines/
│   └── stego_diffusion_pipeline.py # End-to-end embedding & extraction orchestrator
├── scripts/
│   ├── verify_reversibility.py    # Automated reversibility & visual fidelity benchmarks
│   └── benchmark_steganalysis.py  # Statistical normality & steganalysis tests
├── tests/
│   └── test_stego_suite.py        # Comprehensive unit & integration tests
├── output/benchmarks/             # Benchmark reports, CSV results, and residual heatmaps
├── requirements.txt               # Dependencies (PyTorch >= 2.1, SciPy, Diffusers, etc.)
└── LICENSE                        # MIT License
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Siddhantbht02/Message-What-Message-.git
cd Message-What-Message-
pip install -r requirements.txt
```

### 2. Run Test Suite

Verify all mathematical invariants, error-correction tolerances, and pipeline reversibility:

```bash
python -m unittest tests/test_stego_suite.py -v
```

### 3. Run Reversibility Benchmark

Run 20 automated trials at 50 DDIM steps to evaluate visual fidelity (PSNR, SSIM, LPIPS) and extraction recovery (0.00% BER):

```bash
python scripts/verify_reversibility.py --runs 20 --steps 50
```

Results and comparison panels with amplified residual heatmaps are saved to `output/benchmarks/`.

### 4. Run Steganalysis Statistical Benchmark

Evaluate empirical distribution invariance against Kolmogorov-Smirnov and Anderson-Darling tests:

```bash
python scripts/benchmark_steganalysis.py --samples 50
```

---

## Python API Usage

```python
import torch
from pipelines.stego_diffusion_pipeline import StegoDiffusionPipeline
from inversion.fixed_point_ddim import FixedPointDDIMInverter
from codec.distribution_preserving import DistributionPreservingCodec
from codec.ecc_engine import ECCEngine

# Initialize pipeline with diffusion model and beta schedule
pipeline = StegoDiffusionPipeline(
    model=model,
    betas=betas,
    image_shape=(3, 64, 64),
)

# 1. Embed Secret Message
secret_message = b"Top Secret Payload 2026"
stego_image, metrics = pipeline.embed(
    secret_message=secret_message,
    steps=50,
    ecc_rate="1/2",
)

print(f"Embedding successful! PSNR: {metrics.psnr:.2f} dB, BER: {metrics.ber:.4f}")

# 2. Extract Secret Message Autonomously
recovered_message, is_valid = pipeline.extract(stego_image, steps=50)
assert is_valid and recovered_message == secret_message
print("Extraction verified:", recovered_message)
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
