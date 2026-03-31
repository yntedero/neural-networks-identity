# Neural Network Identity Function

> A **4-2-4** neural network that learns the identity function for binary vectors using PyTorch.

**Author:** Yurii Ostapchuk

---

## About

This project implements a simple feedforward neural network with Sigmoid activation that learns to reproduce its input at the output, compressing 4-bit binary vectors through a bottleneck of 2 hidden neurons.

```
Input (4) ──> Hidden (2) ──> Output (4)
  [1,0,1,0]   [0.8, 0.2]   [1,0,1,0]
```

## Tasks

| Subtask | Description | Goal |
|---------|-------------|------|
| **1** | Learn identity for **5** selected binary vectors | 100% accuracy |
| **2** | Learn identity for all **16** possible 4-bit vectors | 9-10 correct vectors |

Each subtask includes **3 experiments** with different learning rate strategies:

| # | Strategy | Learning Rate |
|---|----------|---------------|
| 1 | Constant (baseline) | `0.5` |
| 2 | Constant (higher) | `2.0` |
| 3 | Step schedule | `2.0` → `0.5` → `0.05` |

## Results

### Subtask 1 — 5 vectors

| Experiment | Correct | Accuracy | Reliability |
|:-----------|:-------:|:--------:|:-----------:|
| 1.1 (lr=0.5) | 5/5 | 100.0% | 100.0% |
| 1.2 (lr=2.0) | 5/5 | 100.0% | 100.0% |
| **1.3 (step lr)** | **5/5** | **100.0%** | **100.0%** |

### Subtask 2 — 16 vectors

| Experiment | Correct | Accuracy | Reliability |
|:-----------|:-------:|:--------:|:-----------:|
| 2.1 (lr=0.5) | 7/16 | 84.4% | 75.0% |
| 2.2 (lr=2.0) | 9/16 | 87.5% | 65.6% |
| **2.3 (step lr)** | **9/16** | **87.5%** | **81.2%** |

> **Key finding:** Step learning rate schedule (`2.0` → `0.5` → `0.05`) consistently delivers the best results across both subtasks.

## Project Structure

```
identity-ostapchuk/
├── identity-ostapchuk.ipynb   # Jupyter notebook (with discussion)
├── identity-ostapchuk.py      # Python script
├── requirements.txt           # Dependencies
└── models/                    # Saved model weights (.pth)
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run as script
python identity-ostapchuk.py

# Or open in Jupyter
jupyter notebook identity-ostapchuk.ipynb
```

## Requirements

- Python 3.10+
- PyTorch 2.11.0
- NumPy 2.4.2
