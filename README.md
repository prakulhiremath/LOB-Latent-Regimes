# ⚡️ Latent Micro-Regime Early Detection in Limit Order Books

<p align="center">
  <img src="assets/detection.gif" width="850" alt="Detection Timeline Visualization"/>
</p>

<p align="center">
  <a href="#"><img src="[https://img.shields.io/badge/Method-Trigger%20Based-0078D4?style=for-the-badge&logo=lightning](https://img.shields.io/badge/Method-Trigger%20Based-0078D4?style=for-the-badge&logo=lightning)"/></a>
  <a href="#"><img src="[https://img.shields.io/badge/Model-HMM-8A2BE2?style=for-the-badge&logo=scikitlearn](https://img.shields.io/badge/Model-HMM-8A2BE2?style=for-the-badge&logo=scikitlearn)"/></a>
  <a href="#"><img src="[https://img.shields.io/badge/Precision-100%25-D4AF37?style=for-the-badge&logo=target](https://img.shields.io/badge/Precision-100%25-D4AF37?style=for-the-badge&logo=target)"/></a>
  <a href="#"><img src="[https://img.shields.io/badge/Lead--Time-Positive-28A745?style=for-the-badge&logo=clock](https://img.shields.io/badge/Lead--Time-Positive-28A745?style=for-the-badge&logo=clock)"/></a>
  <a href="[https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)"><img src="[https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)"/></a>
</p>

---

## 🔬 Overview

This research investigates whether **latent microstructure dynamics** in Limit Order Books (LOB) can be mathematically identified *before* observable liquidity stress manifests.

Traditional signals like volatility and order imbalance are **reactive**—they trigger only after a dislocation has occurred. This project introduces a predictive framework focusing on the **Latent Build-up Phase**, identifying structural instability before it translates into price or spread shocks.

---

## 🧠 Core Methodology: The Latent Build-up

Market stress is rarely instantaneous; it is preceded by **structural deterioration**. We model this as a three-state latent process:

| State | Regime | Market Description | Signal Characteristic |
| :--- | :--- | :--- | :--- |
| **0** | **Stable** | Balanced liquidity, high resilience | Equilibrium |
| **1** | **Latent Build-up** | Depth erosion, subtle spread drift | **Hidden Instability** |
| **2** | **Stress** | Observable dislocation, price shocks | Reactive |

> **Key Discovery:** A delayed transition from **State 1 → State 2** creates a deterministic prediction window, allowing for early detection with strictly positive lead-time.

---

## 🛠 Detection Framework

The detector employs a high-fidelity fusion of probabilistic and temporal signals to identify the "inflection point" of market health.

### 📡 Signal Integration
* **Probabilistic Instability:** HMM posterior entropy monitoring.
* **Temporal Drift:** Recursive analysis of spread and depth dynamics.
* **Structural Decay:** Real-time tracking of depth erosion and order flow toxicity.

### 🕹 Detection Logic
* **MAX-Trigger Fusion:** Cross-channel integration to capture the first sign of decay.
* **Rising-Edge Detection:** Focusing on the *onset* of change rather than absolute thresholds.
* **Early-Detection Constraint:** Optimization of $\tau < \sigma$, ensuring the signal precedes the event.

---

## 📊 Results & Performance

| Method | Mean $\Delta$ (Lead-Time) | Precision | Coverage |
| :--- | :--- | :--- | :--- |
| **Adaptive Trigger** | **+18.62** | **100%** | 52.6% |
| **Model HMM** | **+14.95** | **100%** | 43.2% |
| **Multi-Trigger** | **+13.15** | **100%** | 28.1% |
| Order Imbalance | -24.84 | 54.9% | 78.7% |
| Volatility | -32.02 | 45.5% | 43.3% |

### Critical Interpretations:
* **Positive Lead-Time:** Our methods detect stress *before* it happens; baselines are strictly negative (lagging).
* **Temporal Validity:** 100% precision indicates zero "false starts" before the latent phase begins.
* **Trade-off:** Coverage levels reflect the conservative nature of high-precision early signals.

---

## 📈 Key Findings
1.  **Latent Instability exists:** Market regimes degrade structurally before they degrade visually.
2.  **Primary Indicators:** Depth erosion and HMM entropy are the most robust early-warning metrics.
3.  **Signal Edge:** Rising-edge detection is essential to bypass the noise inherent in absolute thresholding.
4.  **Performance:** Trigger-based detection consistently outperforms classical econometric baselines.

---

## 📂 Repository Structure

```bash
.
├── experiments/          # Iterative development v1 → v7
├── notebooks/            # Production-grade experiment analysis
├── results/
│   ├── figures/          # High-resolution performance plots
│   └── summary.txt       # Quantified results summary
├── paper/                # Technical manuscript (PDF)
├── assets/               # Visualizations and GIFs
└── README.md
```

---

## 🚀 Reproducibility

Validated across high-compute and local environments:
* **Cloud:** Google Colab (NVIDIA T4)
* **Local:** Apple Silicon (M4 Pro/Max)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-repo/lob-early-detection.git

# Install dependencies
pip install -r requirements.txt

# Execute the final pipeline
python experiments/v7_final.py
```

---

## 📝 Contributions
* **Causal Formulation:** Formalizing the Latent Build-up $\rightarrow$ Stress transition.
* **Temporal Drift:** Identifying subtle drift as a precursor to liquidity voids.
* **MAX Fusion & Rising-Edge:** Novel trigger logic for sub-millisecond microstructure data.
* **Empirical Proof:** Demonstrating strictly positive lead-time over reactive benchmarks.

---

## 📑 Citation

```bibtex
@article{lob_micro_regime_detection_2026,
  title={Early Detection of Latent Micro-Regimes in Limit Order Books},
  author={Hiremath, Prakul},
  year={2026},
  journal={Reproducible Research in Market Microstructure}
}
```

---

<p align="center">
  Built with ☕️ and 🐍 for <b>Reproducible Quantitative Finance.</b>
</p>



⸻

