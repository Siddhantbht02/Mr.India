# Research Paper: Provably Indistinguishable Generative Steganography

This directory contains the complete publication-ready research paper manuscript based on the repository's generative diffusion steganography pipeline, iterative fixed-point DDIM inversion solver, distribution-preserving arithmetic coding (DPAC), and rate-adaptive ECC engine.

---

## Files in this Directory

- [`paper.tex`](paper.tex): Complete IEEE Conference / Transactions LaTeX manuscript.
- [`references.bib`](references.bib): BibTeX bibliography with 24 foundational and state-of-the-art citations.
- [`README.md`](README.md): Compilation and Overleaf upload instructions.

---

## How to Compile

### Option 1: On Overleaf (Recommended)
1. Go to [Overleaf](https://www.overleaf.com/).
2. Click **New Project** $\to$ **Upload Project**.
3. Upload `paper.tex` and `references.bib` (or compress the `paper/` folder into a `.zip` and upload).
4. Set compiler to **pdfLaTeX** or **XeLaTeX**.
5. Click **Recompile**.

### Option 2: Local Compilation (Command Line)

If you have a local TeX distribution (TeX Live, MiKTeX, or MacTeX):

```bash
cd paper

# 1. First pass
pdflatex paper.tex

# 2. Compile bibliography
bibtex paper

# 3. Resolve cross-references
pdflatex paper.tex
pdflatex paper.tex
```

Or using `latexmk`:

```bash
latexmk -pdf paper.tex
```

---

## Paper Abstract & Summary

### Title
**Provably Indistinguishable Generative Steganography via Fixed-Point DDIM Inversion and Distribution-Preserving Latent Embeddings**

### Author
**Siddhant Bhat**  
*Department of Computer Science and Engineering*  
GitHub: [https://github.com/Siddhantbht02/Mr.India](https://github.com/Siddhantbht02/Mr.India)

### Key Novelties Documented:
1. **Iterative Fixed-Point DDIM Solver:** Eliminates continuous-time trajectory discretization drift without intermediate activation caching, driving trajectory divergence below $\tau \le 10^{-6}$ in $K \le 5$ iterations.
2. **Distribution-Preserving Latent Embedder (DPAC):** Proves that keystream-whitened discrete symbols mapped via synchronized dithering and inverse standard normal quantile functions $\Phi^{-1}(u)$ are identically distributed as standard Gaussian $\mathcal{N}(0, \mathbf{I})$.
3. **Autonomous Rate-Adaptive Error Correction:** Systematic linear block coding supporting code rates $R \in \{1/2, 2/3, 3/4, 5/6\}$, burst interleaving, and self-describing 45-byte framing (`MAGIC`, `RATE_ID`, `PAYLOAD_LEN`, `CRC32`, `SHA-256`).
4. **Empirical Benchmarks:**
   - **0.000000% Bit Error Rate (BER)** across 20 trials.
   - **100% Extraction Success** (zero CRC/SHA collisions).
   - **Visual Fidelity:** $\text{PSNR} \ge 45.59$ dB, $\text{SSIM} \ge 0.9999$, $\text{LPIPS} \le 0.0196$.
   - **Steganalysis Resistance:** Kolmogorov-Smirnov $p$-value $= 0.4191 \gg 0.05$ (85% pass rate), Anderson-Darling pass rate $= 90\%$, skewness $= -0.0073$, excess kurtosis $= -0.0053$.
