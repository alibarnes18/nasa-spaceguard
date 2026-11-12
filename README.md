# NASA SpaceGuard AI

> **Independent Machine Learning Project for Spacecraft Telemetry Anomaly Detection**

*Disclaimer: This is an independent engineering and research project utilizing publicly available NASA SMAP and MSL telemetry datasets. It is not an official product of, nor affiliated with or endorsed by, NASA or JPL.*

---

## 📌 Project Overview

NASA SpaceGuard AI is a time-series anomaly detection pipeline designed to identify unexpected behaviors in spacecraft telemetry. Using data from the **Soil Moisture Active Passive (SMAP)** satellite and the **Mars Science Laboratory (MSL / Curiosity)** rover, the project benchmarks statistical baselines against an LSTM-based next-step prediction and dynamic thresholding framework.

The primary engineering philosophy: **Understand → Build → Test → Evaluate → Explain → Publish**.

---

## 🏗️ Project Architecture

```text
nasa-spaceguard-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── data/
│   ├── raw/           # Raw telemetry (.npy, .csv) - excluded from Git
│   └── processed/     # Scaled & windowed datasets - excluded from Git
├── notebooks/         # Exploratory and experimental notebooks
├── src/
│   └── spaceguard/    # Production-ready modular source code
│       ├── data/
│       ├── models/
│       ├── detection/
│       ├── evaluation/
│       └── visualization/
├── scripts/           # Training and inference entry points
├── tests/             # Unit and integration tests
└── reports/
    └── figures/       # Evaluation charts and result tables
```

---

## 🚀 Environment Setup

### 1. Prerequisites
* Python 3.10+ (Current setup: Python 3.13)
* Git
* NVIDIA GPU with CUDA support (Recommended)

### 2. Installation
```powershell
# Clone the repository
git clone <your-repo-url>
cd nasa-spaceguard-ai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.4 support (if GPU is available)
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# Register Jupyter kernel
python -m ipykernel install --user --name=spaceguard-venv --display-name "Python (SpaceGuard .venv)"
```

---

## 🗺️ Project Roadmap

- [x] **Stage 0: Environment Setup & Scaffolding**
- [x] **Stage 1: Data Engineering & Exploratory Data Analysis (EDA)**
- [x] **Stage 2: Statistical Baseline (Z-score & Event Scoring)**
- [x] **Stage 3: LSTM Prediction Model**
  - Designed dynamic input dimension multi-layer LSTM architecture in `src/spaceguard/models/lstm.py`.
  - Implemented 250-timestep sliding window and chronological train/val split in `src/spaceguard/data/preprocessing.py`.
  - Developed GPU-accelerated trainer with MSE loss, Adam optimizer, early stopping, and checkpointing in `src/spaceguard/models/trainer.py`.
  - Trained baseline LSTM model on `P-1` channel using RTX 3050 CUDA acceleration (Validation Loss: 0.0269).
- [ ] **Stage 4: Dynamic Thresholding & Anomaly Evaluation**
- [ ] **Stage 5: API & Dashboard Deployment**
- [ ] **Stage 6: Final Technical Report & Release**