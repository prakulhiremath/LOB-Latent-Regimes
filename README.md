# ⚡️ Latent Micro-Regime Early Detection in LOBs

<p align="center">
  <img src="assets/detection.gif" width="800" alt="Latent Detection Demo"/>
</p>

<p align="center">
  <img src="[https://img.shields.io/badge/Status-Cutting--Edge-brightgreen?style=for-the-badge](https://img.shields.io/badge/Status-Cutting--Edge-brightgreen?style=for-the-badge)" alt="Status"/>
  <img src="[https://img.shields.io/badge/Framework-HMM--Trigger-blueviolet?style=for-the-badge](https://img.shields.io/badge/Framework-HMM--Trigger-blueviolet?style=for-the-badge)" alt="Framework"/>
  <img src="[https://img.shields.io/badge/Hardware-NVIDIA%20T4%20%7C%20Apple%20M4-black?style=for-the-badge](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20%7C%20Apple%20M4-black?style=for-the-badge)" alt="Hardware"/>
  <img src="[https://img.shields.io/badge/Precision-100%25-orange?style=for-the-badge](https://img.shields.io/badge/Precision-100%25-orange?style=for-the-badge)" alt="Precision"/>
</p>

---

## 🌪 The Thesis: Stress is not a Bolt from the Blue
Standard market signals like **Volatility** and **Imbalance** are *post-mortem* indicators—they tell you the ship is sinking while you're already underwater. 

This project proves that **liquidity instability has a "incubation period."** We capture the latent structural decay *before* the price cracks.

> **"If you wait for the volatility spike, you've already lost. We detect the silence before the scream."**

---

## 🧠 The Architecture: Causal Delayed Transition
We move beyond simple binary states. Our model architecture assumes a three-stage causal evolution of market failure:

| State | Regime | Market Reality | Observability |
| :--- | :--- | :--- | :--- |
| **🟢 0** | **Stable** | Equilibrium liquidity | High |
| **🟡 1** | **Latent Build-up** | **Hidden deterioration / Depth erosion** | **Invisible (The "Ghost" Phase)** |
| **🔴 2** | **Stress** | Observable Dislocation / Crash | High (Reactive) |

### 🛠 The "God-Mode" Detection Logic
Our detector operates on a **MAX-trigger fusion** across high-dimensional channels:
* **Probabilistic Instability:** HMM posterior entropy spikes.
* **Temporal Drift:** Recursive analysis of spread & depth velocity.
* **Structural Erosion:** Real-time order flow decay signals.
* **Rising-Edge Trigger:** Precise identification of the *onset* of instability.

---

## 📊 Performance Matrix (The "Kill-The-Baselines" Table)

While classical signals lag by **20+ units**, our **Adaptive Trigger** provides an average lead-time of **+18.62**, catching the regime shift before it manifests in price.

| Method | Mean $\Delta$ (Lead-Time) | Precision | Coverage |
| :--- | :--- | :--- | :--- |
| **🔥 Adaptive Trigger** | **+18.62** | **100%** | **52.6%** |
| 🧬 Model-Only | +14.95 | 100% | 43.2% |
| 📡 Multi-Trigger | +13.15 | 100% | 28.1% |
| 📉 Imbalance | -24.84 | 54.9% | 78.7% |
| 📉 Volatility | -32.02 | 45.5% | 43.3% |

---

## 📂 Project Anatomy
```bash
.
├── 🧪 experiments/      # Evolutionary stages v1 → v7 (The Lab)
├── 📓 notebooks/        # Final production-grade experiment notebook
├── 📈 results/          # High-fidelity figures & summary.txt
├── 📄 paper/            # The theoretical foundation (PDF)
├── 🎨 assets/           # Dynamic visualization & GIFs
└── 📜 README.md         # You are here.
```

---

## ⚡ Quick Start (Reproduction Pipeline)

Tested on **NVIDIA T4 (Colab)** and **Apple Silicon M4**. Results are 100% deterministic via seeded initialization.

```bash
# Clone the madness
git clone https://github.com/your-repo/latent-lob-detection.git
cd latent-lob-detection

# Arm the environment
pip install -r requirements.txt

# Run the flagship pipeline
python experiments/v7_final.py
```

---

## 🏆 Key Contributions
1.  **Causal Latent Theory:** Formalized the $Regime 1 \rightarrow Regime 2$ transition delay.
2.  **Zero-False-Positive Logic:** 100% precision thresholding in the early-detection window.
3.  **The Trigger Fusion:** Combined entropy, drift, and flow into a single "Early-Warning" signal.
4.  **Positive Lead-Time:** Mathematically demonstrated dominance over reactive baselines.

---

## 📝 Citation
```bibtex
@article{lob_micro_regime_detection_2026,
  title={Early Detection of Latent Micro-Regimes in Limit Order Books},
  author={Hiremath, Prakul},
  year={2026}
}
```

---

<p align="center">
  Built with ☕️, 🐍, and a relentless pursuit of Alpha. <br/>
  <b>License: MIT</b>
</p>
