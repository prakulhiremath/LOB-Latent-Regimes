# Reproducibility Guide

This repository contains the complete pipeline for generating the synthetic Limit Order Book (LOB) data, detection signals, and evaluation metrics presented in the paper.

## 💻 Environment & Hardware
Experiments were conducted across the following environments:
* **Primary Runtime:** Google Colab (NVIDIA T4 GPU)
* **Local Testing:** Apple MacBook (M4 Architecture)

> [!NOTE]
> All experiments are fully deterministic using specified random seeds. Minor variations may occur due to differences in hardware floating-point precision or library versions.

## 🛠 Prerequisites
Ensure you have **Python 3.x** installed. The core logic relies on the following stack:
* `NumPy` & `SciPy` (Numerical processing)
* `scikit-learn` (Evaluation metrics)
* `hmmlearn` (Hidden Markov Modeling)

## 🚀 Quick Start
To reproduce the paper's results, figures, and tables:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Execute Pipeline:**
    Run the notebook cells sequentially. The pipeline is designed to be executed end-to-end to generate:
    * **Synthetic LOB Data:** Simulated market depth and flow.
    * **Detection Signals:** Primary output of the proposed model.
    * **Metrics & Figures:** All visualizations and tables used in the final publication.

---
