# 🥷 Mr.India
**Upgraded Generative Diffusion Steganography Pipeline**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)

A high-capacity, statistically undetectable generative steganography pipeline based on diffusion models (DDIM). This upgraded architecture eliminates discretization drift between inversion and generation trajectories, guarantees **0.00% Bit Error Rate (BER)** with rate-adaptive error correction, and enforces exact distribution invariance against modern steganalysis.

---

## 📑 Table of Contents
- [Pipeline Architecture](#-pipeline-architecture)
- [Key Features](#-key-features)
- [Directory Structure](#-directory-structure)
- [Quickstart](#-quickstart)
  - [Installation](#1-installation)
  - [Run Tests](#2-run-test-suite)
  - [Benchmarks](#3-run-reversibility-benchmark)
- [API Usage](#-python-api-usage)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🏗 Pipeline Architecture

The pipeline securely embeds secret messages into generated images by modifying the latent noise $x_T$ and utilizing fixed-point DDIM inversion for exact reversibility.

```mermaid
graph LR
    subgraph Sender [Embedding Phase]
        M[Secret Message] -->|ECC + Framing| C(Coded Bits)
        C -->|DPAC Embedder| Z[Latent $z_{stego}$]
        Z -->|DDIM Generation| S(Stego Image)
    end
    
    subgraph Receiver [Extraction Phase]
        S2(Stego Image) -->|Fixed-Point DDIM Inversion| Z2[Latent $z'_{stego}$]
        Z2 -->|DPAC Decoder| C2(Extracted Bits)
        C2 -->|ECC Decode| M2[Recovered Message]
    end
    
    S -.->|Public Channel| S2
```

---

## ✨ Key Features

- 🔄 **Exact Fixed-Point DDIM Inversion Solver (`inversion/`):**
  Eliminates numerical trajectory drift without caching all intermediate steps using an iterative fixed-point formulation with bidirectional consistency checks ($K=5, \tau \le 10^{-6}$) and FP32 arithmetic upcasting.

- 🧮 **Distribution-Preserving Latent Bit Embedder (`codec/`):**
  Implements Distribution-Preserving Arithmetic Coding (DPAC) with synchronized keystream whitening and inverse Gaussian CDF transformation $\Phi^{-1}(u)$, guaranteeing zero statistical divergence against Kolmogorov-Smirnov and Anderson-Darling tests ($p > 0.05$).

- 🛡️ **Rate-Adaptive Error Correction Engine (`codec/`):**
  Provides systematic linear block coding supporting code rates $R \in \{1/2, 2/3, 3/4, 5/6\}$, burst error dispersing interleaving, and autonomous self-describing framing (`[ MAGIC | RATE_ID | PAYLOAD_LEN | CRC32 | SHA256 ]`).

- 🚀 **Unified Stego Diffusion Pipeline (`pipelines/`):**
  Clean, high-level API for end-to-end secret embedding and extraction with real-time fidelity tracking (PSNR $\ge 45\text{ dB}$, SSIM $\ge 0.999$, LPIPS $< 0.02$).

- 📊 **Automated Verification & Steganalysis Suite (`scripts/` & `tests/`):**
  Automated multi-regime evaluation scripts with residual difference heatmaps, structured CSV benchmarking, and unit test suites.

---

## 📂 Directory Structure

```text
Mr.India/
├── inversion/
│   ├── base_inversion.py          # Abstract inversion interface & alpha scheduling
│   └── fixed_point_ddim.py        # Iterative fixed-point DDIM inversion solver
├── codec/
│   ├── ecc_engine.py              # Rate-adaptive ECC engine & autonomous framing
│   └── distribution_preserving.py # Distribution-Preserving Arithmetic Coding (DPAC)
├── pipelines/
│   └── stego_diffusion_pipeline.py# End-to-end embedding & extraction orchestrator
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

## 🚀 Quickstart

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Siddhantbht02/Mr.India.git
cd Mr.India
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
> **Note:** Results and comparison panels with amplified residual heatmaps are saved to `output/benchmarks/`.

### 4. Run Steganalysis Statistical Benchmark

Evaluate empirical distribution invariance against Kolmogorov-Smirnov and Anderson-Darling tests:

```bash
python scripts/benchmark_steganalysis.py --samples 50
```

---

## 💻 Python API Usage

The pipeline exposes a unified API that seamlessly handles message encoding, DDIM generation, and extraction.

```python
import torch
from pipelines.stego_diffusion_pipeline import StegoDiffusionPipeline

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

## 🙏 Acknowledgements

This implementation is based on and inspired by:
- **Denoising Diffusion Implicit Models (DDIM)** by Jiaming Song, Chenlin Meng and Stefano Ermon ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502))
- [Ho et al., DDPM TensorFlow repo](https://github.com/hojonathanho/diffusion)
- [ncsnv2 Code Structure](https://github.com/ermongroup/ncsnv2)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
