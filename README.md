ad# Neural Network Identity Function

A 4-2-4 neural network that learns the identity function for binary vectors using PyTorch.

**Author:** Yurii Ostapchuk

## About

This project implements a simple feedforward neural network with topology **4 -> 2 -> 4** and Sigmoid activation. The network learns to reproduce its input at the output, compressing 4-bit binary vectors through a bottleneck of 2 hidden neurons.

### Tasks

- **Subtask 1** - Learn identity for 5 selected binary vectors
- **Subtask 2** - Learn identity for all 16 possible 4-bit binary vectors

Each subtask includes 3 experiments with different learning rate strategies:
1. Constant lr = 0.5
2. Higher lr = 2.0
3. Step schedule (2.0 -> 0.5 -> 0.05)

## Project structure

```
identity-ostapchuk/
  identity-ostapchuk.ipynb   # Jupyter notebook
  identity-ostapchuk.py      # Python script
  requirements.txt           # Dependencies
  models/                    # Saved model weights (.pth)
```

## How to run

```bash
pip install -r requirements.txt
python identity-ostapchuk.py
```

Or via Jupyter:

```bash
jupyter notebook identity-ostapchuk.ipynb
```

## Results

| Experiment | Correct vectors | Accuracy | Reliability |
|---|---|---|---|
| 1.1 (lr=0.5) | 5/5 | 100.0% | 100.0% |
| 1.2 (lr=2.0) | 5/5 | 100.0% | 100.0% |
| 1.3 (step lr) | 5/5 | 100.0% | 100.0% |
| 2.1 (lr=0.5) | 7/16 | 84.4% | 75.0% |
| 2.2 (lr=2.0) | 9/16 | 87.5% | 65.6% |
| 2.3 (step lr) | 9/16 | 87.5% | 81.2% |

## Requirements

- Python 3.10+
- PyTorch 2.11.0
- NumPy 2.4.2
