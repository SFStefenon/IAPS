gpu_use = False
gpu_number = "1"
if gpu_use == True:
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda:0")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(f"GPU {gpu_number}")

complete = True
if complete: # Complete experiment
    EPOCHS = 500
    RUNS = 50
else:
    EPOCHS = 2
    RUNS = 2

import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import math
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOOKBACK = 512
HORIZON = 7
PATCH_SIZE = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIFFUSION_STEPS = 100
DIFFUSION_WEIGHT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

from pathlib import Path
OUTPUT_DIR = Path("Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("tucurui.csv", sep=";", decimal=",")
data = df[["Natural Flow", "UPH610010000"]].dropna()
values = data["Natural Flow"].to_numpy(dtype=np.float32)
precipitation = data["UPH610010000"].to_numpy(dtype=np.float32)
samples = np.arange(len(values))
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(samples, values, color="k", linewidth=1, label="Natural flow")
ax1.set_xlabel("Time step")
ax1.set_ylabel(r"Natural flow ($\mathrm{m^3/s}$)", color="k")
ax1.tick_params(axis="y", labelcolor="k")
ax1.set_xlim(0, len(values) - 1)
ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(samples, precipitation, color="green", linewidth=1, alpha=0.6, label="Precipitation")
ax2.set_ylabel("Precipitation (mm)", color="green")
ax2.tick_params(axis="y", labelcolor="green")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tucurui_time_series.pdf", bbox_inches="tight")
plt.show()

# ============================================================
# Second plot: normalization and train/test split
# ============================================================

split = int(len(values) * 0.8)

# Calculate normalization statistics using only training data
mean = values[:split].mean()
std = values[:split].std()

normalized_values = (values - mean) / std

train_values = normalized_values[:split]
test_values = normalized_values[split - LOOKBACK:]

train_samples = samples[:split]
test_samples = samples[split:]
lookback_samples = samples[split - LOOKBACK:split]

fig, ax = plt.subplots(figsize=(10, 4))

# Training set
ax.plot(
    train_samples,
    train_values,
    color="black",
    linewidth=1,
    label="Training set (80%)",
    zorder=3,
)

# Lookback observations included in the test windows
ax.plot(
    lookback_samples,
    normalized_values[split - LOOKBACK:split],
    color="orange",
    linewidth=1.5,
    label=f"Test lookback ({LOOKBACK} steps)",
    zorder=4,
)

# Actual test period
ax.plot(
    test_samples,
    normalized_values[split:],
    color="red",
    linewidth=1,
    label="Test set (20%)",
    zorder=3,
)

# Train/test boundary
ax.axvline(
    split,
    color="green",
    linestyle="--",
    linewidth=1.5,
    label="Train/test split",
    zorder=5,
)

# Highlight the lookback interval
ax.axvspan(
    split - LOOKBACK,
    split,
    color="orange",
    alpha=0.15,
    zorder=1,
)

# Zero represents the training-set mean after normalization
ax.axhline(
    0,
    color="gray",
    linestyle=":",
    linewidth=1,
    zorder=2,
)

ax.set_xlabel("Time step")
ax.set_ylabel("Normalized natural flow")
ax.set_xlim(0, len(normalized_values) - 1)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "tucurui_normalization_split.pdf",
    bbox_inches="tight",
)
plt.show()

print(f"Total samples: {len(values)}")
print(f"Training samples: {len(train_values)}")
print(f"Test-period samples: {len(normalized_values) - split}")
print(f"Test array including lookback: {len(test_values)}")
print(f"Training mean: {mean:.4f}")
print(f"Training standard deviation: {std:.4f}")

