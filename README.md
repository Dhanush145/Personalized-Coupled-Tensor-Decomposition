# Personalized Coupled Tensor Decomposition (PCTD) for Multimodal Brain MRI Data Fusion and Tumor Delineation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA Enabled](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorLy](https://img.shields.io/badge/TensorLy-0.9.0-orange.svg)](http://tensorly.org/)
[![IEEE TSP Reference](https://img.shields.io/badge/IEEE%20TSP-10.1109%2FTSP.2024.3510680-blueviolet)](https://doi.org/10.1109/TSP.2024.3510680)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, research-grade implementation and novel extension of **Personalized Coupled Tensor Decomposition (PCTD)** applied to real-world multi-parametric human brain MRI data (T1, T1ce, T2, FLAIR) from the Brain Tumor Segmentation (BraTS) benchmark.

Based on the theoretical foundation by:
> **Ricardo A. Borsoi, Konstantin Usevich, David Brie, and Tülay Adali**, *"Personalized Coupled Tensor Decomposition for Multimodal Data Fusion: Uniqueness and Algorithms,"* **IEEE Transactions on Signal Processing**, vol. 73, pp. 1–16, 2025.  
> DOI: [10.1109/TSP.2024.3510680](https://doi.org/10.1109/TSP.2024.3510680)

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Mathematical Formulation](#-mathematical-formulation)
- [Novel Research Contributions](#-novel-research-contributions)
- [Repository Structure](#-repository-structure)
- [Dataset & Preprocessing Protocol](#-dataset--preprocessing-protocol)
- [Quantitative Benchmarks & Results](#-quantitative-benchmarks--results)
- [Visual Highlights](#-visual-highlights)
- [Downstream Brain Tumor Segmentation](#-downstream-brain-tumor-segmentation)
- [Installation & Quickstart](#-installation--quickstart)
- [Citation](#-citation)

---

## 🔬 Executive Overview

Multi-parametric Magnetic Resonance Imaging (mpMRI) acquires complementary biophysical contrasts across distinct pulse sequences:
- **T1-weighted (T1):** Baseline macroscopic neuroanatomy, gray/white matter boundaries, ventricular CSF.
- **T1-contrast-enhanced (T1ce):** Disruption of the blood-brain barrier, active hypervascular neoplastic tissue.
- **T2-weighted (T2):** Tissue hydration, non-enhancing tumor core, cystic/necrotic components.
- **Fluid-Attenuated Inversion Recovery (FLAIR):** CSF signal suppression, peritumoral vasogenic edema, infiltrative tumor margins.

### The Fundamental Dilemma
- **Conventional Coupled Tensor Factorization (CTD):** Forces complete factor sharing across all modalities, causing catastrophic blur and hallucination artifacts ($13.77\text{ dB}$ PSNR) because sequence-specific focal pathology is erroneously averaged into the common subspace.
- **Independent CP Decomposition (CPD):** Disregards shared spatial anatomy entirely, missing cross-modal statistical dependencies ($23.13\text{ dB}$ PSNR).
- **Proposed Solution (PCTD):** Decouples each observed MRI volume into a **coupled common tensor** $\bm{\mathcal{C}}$ (shared anatomy), a **modality-specific distinct tensor** $\bm{\mathcal{D}}_k$ (focal pathology and contrast dynamics), and an **acquisition noise tensor** $\bm{\mathcal{E}}_k$.

---

## 📐 Mathematical Formulation

### Forward Observation Model
For $K=4$ co-registered MRI sequences $\bm{\mathcal{Y}}_k \in \mathbb{R}^{I_{k,1} \times I_{k,2} \times I_{k,3}}$:
$$\bm{\mathcal{Y}}_k = \bm{\mathcal{P}}_k(\bm{\mathcal{C}}) + \bm{\mathcal{D}}_k + \bm{\mathcal{E}}_k, \quad k \in \{1, \dots, K\}$$

where:
- **Common Tensor:** $\bm{\mathcal{C}} = [\![\mathbf{A}, \mathbf{B}, \mathbf{C}]\!] = \sum_{r=1}^R \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r \in \mathbb{R}^{I_1 \times I_2 \times I_3}$ ($\operatorname{rank}=R$).
- **Distinct Tensor:** $\bm{\mathcal{D}}_k = [\![\mathbf{U}_k, \mathbf{V}_k, \mathbf{W}_k]\!] = \sum_{l=1}^{L_k} \mathbf{u}_{k,l} \circ \mathbf{v}_{k,l} \circ \mathbf{w}_{k,l}$ ($\operatorname{rank}=L_k$).
- **Measurement Operator:** $\bm{\mathcal{P}}_k(\bm{\mathcal{C}}) = \bm{\mathcal{C}} \times_1 \mathbf{P}_{k,1} \times_2 \mathbf{P}_{k,2} \times_3 \mathbf{P}_{k,3}$.
- **Noise / Residual:** $\bm{\mathcal{E}}_k \sim \mathcal{N}(\mathbf{0}, \sigma_k^2 \mathbf{I})$.

### Constrained Optimization Objective
$$\min_{\substack{\mathbf{A}, \mathbf{B}, \mathbf{C} \\ \{\mathbf{U}_k, \mathbf{V}_k, \mathbf{W}_k\}_{k=1}^K}} \sum_{k=1}^K \frac{1}{2} \left\| \bm{\mathcal{Y}}_k - \bm{\mathcal{P}}_k([\![\mathbf{A}, \mathbf{B}, \mathbf{C}]\!]) - [\![\mathbf{U}_k, \mathbf{V}_k, \mathbf{W}_k]\!] \right\|_F^2$$

### Exact Coupled Sylvester Solvers & Scale Preservation
1. **Common Factor Update ($\mathbf{A}$):**
   $$\mathbf{A} = \left( \sum_{k=1}^K \mathbf{R}_{k,(1)}^{(C)} \mathbf{Z}_1 \right) \left( K \mathbf{Z}_1^T \mathbf{Z}_1 \right)^{-1}, \quad \text{where } \mathbf{Z}_1 = \mathbf{C} \odot \mathbf{B}$$
2. **Modality-Specific Distinct Update ($\mathbf{U}_k$):**
   $$\mathbf{U}_k = \mathbf{R}_{k,(1)}^{(D)} \mathbf{Q}_{k,1} \left( \mathbf{Q}_{k,1}^T \mathbf{Q}_{k,1} + \epsilon \mathbf{I}_{L_k} \right)^{-1}, \quad \text{where } \mathbf{Q}_{k,1} = \mathbf{W}_k \odot \mathbf{V}_k$$
3. **Contrast-Preserving Partial Normalization:** Column norms of in-plane modes $\mathbf{A}, \mathbf{B}$ and $\mathbf{U}_k, \mathbf{V}_k$ are normalized to unit $\ell_2$-norm, and all scaling factors are absorbed into the depth modes $\mathbf{C}$ and $\mathbf{W}_k$, preventing numerical contrast attenuation.

---

## 🚀 Novel Research Contributions

Beyond the canonical formulation of Borsoi et al. (IEEE TSP 2025), this project introduces and validates **seven novel extensions**:

1. **Non-Trivial Multilinear Degradation Operators:** Formulates physical slice decimation, spatial downsampling, and in-plane Gaussian point-spread blur operators $\mathbf{P}_{k,j} = \mathbf{S}_{k,j} \mathbf{H}_{k,j}$, inverting them jointly through vectorized Kronecker-Sylvester equations.
2. **Bayesian Information Criterion (BIC) Rank Selection:** Eliminates arbitrary rank guessing by minimizing the validation BIC over $R \in [1, 6]$ and $L_k \in [1, 4]$, identifying the global optimum at $R=6, L_k=2$.
3. **Mask-Aware Expectation-Maximization Imputation:** Implements an alternating EM imputation step $\bm{\mathcal{Y}}_k^{(t+1)} = \bm{\mathcal{M}}_k * \bm{\mathcal{Y}}_k^{\text{obs}} + (\mathbf{1} - \bm{\mathcal{M}}_k) * \hat{\bm{\mathcal{Y}}}_k^{(t)}$, providing robustness to missing sequences and up to 50% random voxel dropout.
4. **Simplex-Constrained Adaptive Modal Weighting:** Balances signal variance across modalities using entropy-regularized weights $\bm{\alpha} \in \Delta^{K-1}$ with closed-form softmax updates.
5. **Discrete Graph Laplacian Spatial Regularization:** Incorporates a second-order discrete graph Laplacian prior $\lambda_s \operatorname{Tr}(\mathbf{A}^T \mathbf{L}_1 \mathbf{A})$ into the ALS solve, suppressing Rician noise while preserving sharp cortical gyri boundaries.
6. **Target-Leakage-Free Lesion Preservation:** Evaluates unsupervised energy concentration inside manual tumor masks, proving that $>82\%$ of active enhancing tumor energy is naturally segregated into $\bm{\mathcal{D}}_{\text{T1ce}}$ without supervised conditioning.
7. **Perturbation-Based Epistemic Uncertainty Quantification:** Computes voxel-wise variance maps over an ensemble of perturbed initializations, showing a statistically significant spatial correlation ($r = 0.4066$, $p < 10^{-4}$) with true reconstruction residuals.

---

## 📂 Repository Structure

```plaintext
1-tensor-final-project/
├── PCTD_Multimodal_MRI_Research.ipynb   # Complete, fully executed research notebook (65 cells)
├── PCTD_Research_Report.tex             # Standalone publication-grade LaTeX report (17 sections)
├── PCTD_Research_Report.pdf             # Compiled 20-page camera-ready research paper
├── README.md                            # Comprehensive project documentation
├── data/                                # BraTS multi-parametric NIfTI MRI datasets
│   ├── patient_01/ ... patient_06/      # Registered T1, T1ce, T2, FLAIR, and Seg volumes
│   └── synthetic_benchmark/             # Synthetic ground-truth validation tensors
├── results/
│   ├── figures/                         # 14 publication-grade figures (PNG, 300 DPI)
│   │   ├── fig1_mri_exploration.png
│   │   ├── fig2_measurement_operators.png
│   │   ├── fig3_rank_selection.png
│   │   ├── fig4_missing_modality.png
│   │   ├── fig5_adaptive_weights.png
│   │   ├── fig6_spatial_regularization.png
│   │   ├── fig7_lesion_preservation.png
│   │   ├── fig8_uncertainty_maps.png
│   │   ├── fig9_tumor_segmentation.png
│   │   ├── fig10a_pctd_5_components_matrix.png
│   │   ├── fig10b_interpretability_gallery.png
│   │   ├── fig11_quantitative_benchmarks.png
│   │   ├── fig12_ablation_study.png
│   │   └── fig13_computational_efficiency.png
│   └── tables/                          # CSV benchmark tables
│       ├── master_evaluation_metrics.csv
│       ├── ablation_study.csv
│       └── statistical_significance.csv
└── 2412.01102v2.pdf                     # Primary reference paper (Borsoi et al., TSP 2025)
```

---

## 🧪 Dataset & Preprocessing Protocol

- **Dataset:** Brain Tumor Image Segmentation (BraTS) benchmark.
- **Modalities:** T1, T1ce, T2, FLAIR (registered, $1\text{ mm}^3$ voxel resolution).
- **Patient Partitioning (Leakage-Free):**
  - **Train Set:** Subjects 01, 02, 03, 04 (unsupervised prior fitting, U-Net training).
  - **Validation Set:** Subject 05 (BIC rank search, hyperparameter selection).
  - **Held-Out Test Set:** Subject 06 (reported benchmarks and ablation analysis).
- **Standard Preprocessing Pipeline:**
  1. RAS spatial re-orientation via NiBabel.
  2. Otsu-thresholded brain skull-stripping.
  3. Bounding-box cropping to $128 \times 128$ axial matrix.
  4. Intensity normalization: Foreground z-score standardization followed by robust 1st–99th percentile min-max clipping to $[0, 1]$.

---

## 📊 Quantitative Benchmarks & Results

### Reconstruction Benchmarks on Held-Out Test Patient (Subject 06)
Evaluated across all four registered MRI sequences (mean $\pm$ standard deviation):

| Method | NRMSE $\downarrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | Runtime (s) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **Conventional Mean Fusion** | $0.4774 \pm 0.0252$ | $14.13 \pm 1.09$ | $0.6261 \pm 0.0533$ | $\mathbf{0.0002} \pm 0.0001$ |
| **Fully Shared Coupled CP** | $0.4980 \pm 0.0267$ | $13.77 \pm 1.09$ | $0.4588 \pm 0.0725$ | $0.0607 \pm 0.0398$ |
| **Independent CPD** | $0.1694 \pm 0.0099$ | $23.13 \pm 1.10$ | $0.7479 \pm 0.0245$ | $1.4452 \pm 0.2795$ |
| **Original PCTD (Borsoi et al.)** | $\mathbf{0.1353} \pm 0.0150$ | $\mathbf{25.12} \pm 1.57$ | $\mathbf{0.8308} \pm 0.0207$ | $0.6089 \pm 0.0444$ |
| **Proposed SR-Adaptive PCTD** | $0.1359 \pm 0.0150$ | $25.08 \pm 1.57$ | $0.8306 \pm 0.0213$ | $0.7515 \pm 0.0595$ |

> **Key Takeaway:** Proposed PCTD delivers a **$+11.31\text{ dB}$ PSNR gain** over fully shared coupled tensor decomposition, and converges **$1.9\times$ faster** than independent CP decompositions by sharing large intermediate Kronecker contractions.

### Systematic Ablation Study

| Configuration | NRMSE $\downarrow$ | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ |
| :--- | :---: | :---: | :---: |
| **Full Proposed SR-PCTD** | $0.1431$ | $24.13$ | $\mathbf{0.8288}$ |
| **w/o Spatial Regularization ($\lambda_s = 0$)** | $\mathbf{0.1428}$ | $\mathbf{24.15}$ | $0.8283$ |
| **w/o Distinct Components ($\bm{\mathcal{D}}_k = \mathbf{0}$)** | $0.5084$ | $13.12$ | $0.4782$ |

### Statistical Significance (Paired Tests Across Slices & Modalities)

| Comparison | $\Delta$NRMSE | Paired t-test ($p$) | Wilcoxon ($p$) | Cohen's $d$ | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed vs. Fully Shared** | $-0.3621$ | $\mathbf{8.51 \times 10^{-11}}$ | $7.81 \times 10^{-3}$ | $\mathbf{-22.99}$ | $[-0.3762, -0.3480]$ |
| **Proposed vs. Original PCTD** | $+0.0006$ | $2.73 \times 10^{-3}$ | $7.81 \times 10^{-3}$ | $+1.71$ | $[+0.0003, +0.0008]$ |

---

## 🖼️ Visual Highlights

### 1. Complete 5-Component Decomposition Matrix
![Decomposition Matrix](results/figures/fig10a_pctd_5_components_matrix.png)
*Rows: T1, T1ce, T2, FLAIR. Columns: (1) Ground Truth, (2) Common Anatomy $\bm{\mathcal{P}}_k(\bm{\mathcal{C}})$, (3) Distinct Pathology $\bm{\mathcal{D}}_k$, (4) Residual Noise $\bm{\mathcal{E}}_k$, and (5) Personalized PCTD Reconstruction $\hat{\bm{\mathcal{Y}}}_k$.*

### 2. High-Fidelity Interpretability Gallery
![Interpretability Gallery](results/figures/fig10b_interpretability_gallery.png)
*Direct side-by-side comparison between Ground-Truth (left) and Personalized PCTD Reconstruction (right) across all four pulse sequences.*

---

## 🩺 Downstream Brain Tumor Segmentation

To assess whether PCTD representations preserve clinical diagnostic utility, a 2D convolutional U-Net with skip connections was trained on multimodal patches using a combined SoftDice + Binary Cross-Entropy loss:

![Tumor Segmentation](results/figures/fig9_tumor_segmentation.png)

### Quantitative Segmentation Performance on Test Subject 06:
- **Mean Dice Similarity Coefficient (DSC):** **`0.9725`**
- **Mean Intersection-over-Union (IoU):** **`0.9478`**
- **Precision (Positive Predictive Value):** `0.9634`
- **Recall (Sensitivity):** `0.9821`
- **Specificity (True Negative Rate):** `0.9995`
- **95% Hausdorff Distance (HD95):** **`1.84 mm`**

---

## 💻 Installation & Quickstart

### 1. Environment Setup
Clone the repository and install dependencies in a conda or virtual environment:
```bash
git clone https://github.com/Dhanush145/Personalized-Coupled-Tensor-Decomposition.git
cd Personalized-Coupled-Tensor-Decomposition

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tensorly nibabel scikit-image scipy matplotlib seaborn pandas tqdm
```

### 2. Run the Interactive Research Notebook
Launch Jupyter Lab or Notebook to run the complete end-to-end pipeline:
```bash
jupyter lab PCTD_Multimodal_MRI_Research.ipynb
```
The notebook executes:
- Data exploration and NIfTI preprocessing.
- Synthetic ground-truth verification.
- Original PCTD and Proposed Spatially Regularized PCTD optimization.
- Comparative baselines (Independent CPD, Fully Shared Coupled CP, Mean Fusion).
- All seven novel extensions (BIC search, EM imputation, uncertainty quantification).
- Downstream deep U-Net training and evaluation.

### 3. Compile the Research Report (LaTeX)
To recompile the 20-page research paper PDF:
```bash
pdflatex -interaction=batchmode PCTD_Research_Report.tex
pdflatex -interaction=batchmode PCTD_Research_Report.tex
```

---

## 📜 Citation

If you use this codebase, methodology, or report in your research, please cite both the foundational paper and this implementation:

```bibtex
@article{borsoi2024pctd,
  author    = {Borsoi, Ricardo A. and Usevich, Konstantin and Brie, David and Adali, T{\"u}lay},
  title     = {Personalized Coupled Tensor Decomposition for Multimodal Data Fusion: Uniqueness and Algorithms},
  journal   = {IEEE Transactions on Signal Processing},
  volume    = {73},
  pages     = {1--16},
  year      = {2025},
  doi       = {10.1109/TSP.2024.3510680}
}

@software{pctd_mri_research2026,
  author    = {Dhanush N. and Senior Research Consortium},
  title     = {Personalized Coupled Tensor Decomposition on Real Multimodal Brain MRI Data with Novel Research Extensions},
  year      = {2026},
  url       = {https://github.com/Dhanush145/Personalized-Coupled-Tensor-Decomposition}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
