# Neural Network Identity Function

A 4-2-4 neural network that learns the identity function for binary vectors using PyTorch.

## About

This project implements a simple feedforward neural network with topology **4 -> 2 -> 4** and Sigmoid activation. The network learns to reproduce its input at the output, compressing 4-bit binary vectors through a bottleneck of 2 hidden neurons.

### Tasks

- **Task 1** - Learn identity for 5 selected binary vectors
- **Task 2** - Learn identity for all 16 possible 4-bit binary vectors

Each task includes 3 experiments with different learning rate strategies (constant, high, and step schedule).

## How to run

```bash
pip install -r requirements.txt
jupyter notebook neural-identity.ipynb
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Jupyter Notebook
