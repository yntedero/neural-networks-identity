# Task 1 - Identity
# Neural net 4-2-4 with Sigmoid activation.
#
# Author: Ostapchuk

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

print("PyTorch version:", torch.__version__)

# 4-2-4 architecture:
# 4 input neurons -> 2 hidden neurons -> 4 output neurons
# sigmoid squashes everything to [0, 1]


class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 2)
        self.output = nn.Linear(2, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x


print("Network ready")

# training loop
# each epoch shuffles vectors randomly, computes SSE loss per vector,
# and updates weights. returns error history for plotting.


def train(model, data, epochs, lr, print_every=500):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(data))
        total_error = 0.0

        for i in order:
            x = data[i]
            output = model(x)
            loss = torch.sum((output - x) ** 2)
            total_error += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        history.append(total_error)

        if epoch % print_every == 0 or epoch == epochs:
            print(f"Epoch {epoch}  Global error  {total_error:.5f}")

    return history


# testing function
# accuracy = how many bits match after rounding
# [AI] I found the reliability concept via AI - it measures whether outputs
# are decisive (close to 0 or 1) rather than sitting around 0.5


def test(model, data, epsilon=0.2):
    model.eval()
    correct = 0
    total_acc = 0
    total_rel = 0

    print("\nTesting")
    print(
        f"{'Input':<12}{'Output':<12}{'Response':<28}{'Error':<10}{'Accuracy':<12}{'Reliability'}"
    )
    print("-" * 84)

    with torch.no_grad():
        for x in data:
            output = model(x)
            rounded = torch.round(output)
            error = torch.sum((output - x) ** 2).item()

            acc = (rounded == x).sum().item() / len(x) * 100
            total_acc += acc

            rel_ok = ((output < epsilon) | (output > 1 - epsilon)).sum().item()
            rel = rel_ok / len(x) * 100
            total_rel += rel

            if (rounded == x).all():
                correct += 1

            inp = " ".join([str(int(v)) for v in x])
            out = " ".join([str(int(v)) for v in rounded])
            resp = " ".join([f"{v:.2f}" for v in output])
            print(
                f"{inp:<12}{out:<12}{resp:<28}{error:<10.3f}{acc:.0f}%{'':<8}{rel:.0f}%"
            )

    n = len(data)
    print(f"\nCorrect vectors: {correct}/{n}")
    print(f"Average accuracy: {total_acc / n:.1f}%")
    print(f"Average reliability: {total_rel / n:.1f}%")

    model.train()
    return correct, total_acc / n, total_rel / n


# ============================================================
# Subtask 1: Identity for 5 vectors
# The network learns to reproduce 5 hand-picked binary vectors.
# ============================================================

# 5 binary vectors - input should equal output (identity)
data_5 = torch.tensor(
    [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]],
    dtype=torch.float32,
)

print("Data (5 vectors):")
print(data_5)

# Experiment 1.1 - baseline, lr=0.5, 1000 epochs
print("\n=== Experiment 1.1 (lr=0.5, 1000 epochs) ===\n")

torch.manual_seed(42)
model_1_1 = IdentityNet()
hist_1_1 = train(model_1_1, data_5, epochs=1000, lr=0.5, print_every=200)
s_1_1, a_1_1, r_1_1 = test(model_1_1, data_5)

# Experiment 1.2 - higher lr=2.0, more epochs
print("\n=== Experiment 1.2 (lr=2.0, 3000 epochs) ===\n")

torch.manual_seed(42)
model_1_2 = IdentityNet()
hist_1_2 = train(model_1_2, data_5, epochs=3000, lr=2.0, print_every=500)
s_1_2, a_1_2, r_1_2 = test(model_1_2, data_5)

# Experiment 1.3 - step-wise learning rate
# [AI] I got the idea for step-wise lr from AI assistance
# the idea: start with a big lr (fast learning), then gradually
# shrink it (fine-tuning at the end)

print("\n=== Experiment 1.3 (step lr) ===\n")

torch.manual_seed(42)
model_1_3 = IdentityNet()
hist_1_3 = []
print("--- phase 1: lr=2.0, 1000 epochs ---")
h = train(model_1_3, data_5, epochs=1000, lr=2.0, print_every=500)
hist_1_3.extend(h)
print("\n--- phase 2: lr=0.5, 1000 epochs ---")
h = train(model_1_3, data_5, epochs=1000, lr=0.5, print_every=500)
hist_1_3.extend(h)
print("\n--- phase 3: lr=0.05, 500 epochs ---")
h = train(model_1_3, data_5, epochs=500, lr=0.05, print_every=500)
hist_1_3.extend(h)
s_1_3, a_1_3, r_1_3 = test(model_1_3, data_5)

# error plot - subtask 1
plt.figure(figsize=(10, 4))
plt.plot(hist_1_1, label="Exp 1.1 (lr=0.5)")
plt.plot(hist_1_2, label="Exp 1.2 (lr=2.0)")
plt.plot(hist_1_3, label="Exp 1.3 (step lr)")
plt.xlabel("Epoch")
plt.ylabel("Global error")
plt.title("Subtask 1 - Error over training")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_subtask1.png")
plt.show()

# comparison - subtask 1
print("=" * 50)
print("COMPARISON - SUBTASK 1 (5 vectors)")
print("=" * 50)
print(f"Exp 1.1 (lr=0.5):    correct={s_1_1}/5  acc={a_1_1:.1f}%  rel={r_1_1:.1f}%")
print(f"Exp 1.2 (lr=2.0):    correct={s_1_2}/5  acc={a_1_2:.1f}%  rel={r_1_2:.1f}%")
print(f"Exp 1.3 (step lr):   correct={s_1_3}/5  acc={a_1_3:.1f}%  rel={r_1_3:.1f}%")


# ============================================================
# Subtask 2: Identity for 16 vectors
# The network tries to learn all 16 possible 4-bit vectors.
# This is harder because 2 hidden neurons need to encode 16 distinct patterns.
# ============================================================

# generate all 16 binary vectors
# [AI] I looked up the bit-shift approach (>> and &) with AI help
# it works by extracting bits from a number, e.g. 13 (binary 1101) gives [1, 1, 0, 1]
# >> shifts bits right, & 1 grabs the last bit

data_16 = []
for i in range(16):
    vector = [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1]
    data_16.append(vector)

data_16 = torch.tensor(data_16, dtype=torch.float32)

print("\nData (16 vectors):")
for v in data_16:
    print(f"  [{int(v[0])} {int(v[1])} {int(v[2])} {int(v[3])}]")

# Experiment 2.1 - baseline, lr=0.5, 5000 epochs
print("\n=== Experiment 2.1 (lr=0.5, 5000 epochs) ===\n")

torch.manual_seed(42)
model_2_1 = IdentityNet()
hist_2_1 = train(model_2_1, data_16, epochs=5000, lr=0.5, print_every=1000)
s_2_1, a_2_1, r_2_1 = test(model_2_1, data_16)

# Experiment 2.2 - lr=2.0, 10000 epochs
print("\n=== Experiment 2.2 (lr=2.0, 10000 epochs) ===\n")

torch.manual_seed(42)
model_2_2 = IdentityNet()
hist_2_2 = train(model_2_2, data_16, epochs=10000, lr=2.0, print_every=2000)
s_2_2, a_2_2, r_2_2 = test(model_2_2, data_16)

# Experiment 2.3 - step-wise lr, same idea as 1.3
print("\n=== Experiment 2.3 (step lr) ===\n")

torch.manual_seed(42)
model_2_3 = IdentityNet()
hist_2_3 = []
print("--- phase 1: lr=2.0, 5000 epochs ---")
h = train(model_2_3, data_16, epochs=5000, lr=2.0, print_every=2000)
hist_2_3.extend(h)
print("\n--- phase 2: lr=0.5, 3000 epochs ---")
h = train(model_2_3, data_16, epochs=3000, lr=0.5, print_every=2000)
hist_2_3.extend(h)
print("\n--- phase 3: lr=0.05, 2000 epochs ---")
h = train(model_2_3, data_16, epochs=2000, lr=0.05, print_every=2000)
hist_2_3.extend(h)
s_2_3, a_2_3, r_2_3 = test(model_2_3, data_16)

# error plot - subtask 2
plt.figure(figsize=(10, 4))
plt.plot(hist_2_1, label="Exp 2.1 (lr=0.5)")
plt.plot(hist_2_2, label="Exp 2.2 (lr=2.0)")
plt.plot(hist_2_3, label="Exp 2.3 (step lr)")
plt.xlabel("Epoch")
plt.ylabel("Global error")
plt.title("Subtask 2 - Error over training")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_subtask2.png")
plt.show()

# comparison - subtask 2
print("=" * 50)
print("COMPARISON - SUBTASK 2 (16 vectors)")
print("=" * 50)
print(f"Exp 2.1 (lr=0.5):    correct={s_2_1}/16  acc={a_2_1:.1f}%  rel={r_2_1:.1f}%")
print(f"Exp 2.2 (lr=2.0):    correct={s_2_2}/16  acc={a_2_2:.1f}%  rel={r_2_2:.1f}%")
print(f"Exp 2.3 (step lr):   correct={s_2_3}/16  acc={a_2_3:.1f}%  rel={r_2_3:.1f}%")

# save models
os.makedirs("models", exist_ok=True)

torch.save(model_1_1.state_dict(), "models/subtask1_exp1.pth")
torch.save(model_1_2.state_dict(), "models/subtask1_exp2.pth")
torch.save(model_1_3.state_dict(), "models/subtask1_exp3.pth")
torch.save(model_2_1.state_dict(), "models/subtask2_exp1.pth")
torch.save(model_2_2.state_dict(), "models/subtask2_exp2.pth")
torch.save(model_2_3.state_dict(), "models/subtask2_exp3.pth")
print("\nModels saved to models/")

# Takeaways:
#
# Subtask 1 (5 vectors) - the network hits 100% accuracy without much trouble.
# Step-wise lr gives the best reliability.
#
# Subtask 2 (16 vectors) - harder, because 2 hidden neurons have to encode 16 patterns.
# Step-wise lr works best here too.
#
# What I learned:
# - high learning rate = fast training but can get unstable
# - low learning rate = slow but steady
# - step-wise lr (high -> low) is the best strategy overall
