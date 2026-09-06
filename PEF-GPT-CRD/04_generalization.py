"""Final flow-only conditional diffusion LLM selected by ablation and HPO.

The ablation study selected the architecture without precipitation. The
diffusion model remains conditional: it denoises the future-flow residual using
the representation extracted from the historical natural-flow window.

The optimized hyperparameters are those of mTPE-Hyperband tuning.
"""

import os
gpu_use = True
complete = True
generalization_analisys = False

gpu_number = "3"
if gpu_use:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

os.environ.update(
    {
        "USE_TF": "0",
        "USE_FLAX": "0",
        "USE_TORCH": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
)

import gc
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model

warnings.filterwarnings("ignore")

if gpu_use and torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(f"GPU {gpu_number}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Metal Performance Shaders")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Data and experimental settings
if complete:  # Complete experiment
    EPOCHS = 100
    RUNS = 50
else:
    EPOCHS = 2
    RUNS = 2

# Reservoirs used for the generalization analysis:
# AMUHBB - Norte
# AMCLR  - Sudeste/Centro-Oeste
# RIGAR - Sul
# JEUITP - Nordeste
from pathlib import Path

if generalization_analisys:
    OUTPUT_DIR = Path("Results_04_benchmarking_generalization")
    DATA_DIR = Path("dados_hidrologicos")
    RESERVOIR_ID = "IGUHBI"
    FLOW_COLUMN = "val_vazaoafluente"
else:
    OUTPUT_DIR = Path("Results_04_benchmarking")
    DATA_PATH = Path("tucurui.csv")
    FLOW_COLUMN = "Natural Flow"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_RATIO = 0.8
HORIZON = 7
SEEDS = list(range(1, RUNS + 1))
LLM_NAME = "gpt2"
DIFFUSION_SAMPLES = 10

# Best mTPE-Hyperband configuration
LOOKBACK = 64
PATCH_SIZE = 32
DIFFUSION_STEPS = 25
DROPOUT = 0.04253398043289619
DIFFUSION_HIDDEN = 64
DIFFUSION_WEIGHT = 0.029826779854702445
BATCH_SIZE = 32
LEARNING_RATE = 0.000806220692456831
WEIGHT_DECAY = 0.0001258929350911359
GRAD_CLIP = 4.0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_windows(features, target):
    number_of_windows = len(target) - LOOKBACK - HORIZON + 1
    if number_of_windows <= 0:
        raise ValueError("The series is shorter than LOOKBACK + HORIZON.")

    x = np.stack(
        [features[index : index + LOOKBACK] for index in range(number_of_windows)]
    )
    y = np.stack(
        [
            target[index + LOOKBACK : index + LOOKBACK + HORIZON]
            for index in range(number_of_windows)
        ]
    )
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, diffusion_step):
        half = self.dimension // 2
        scale = np.log(10000) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=diffusion_step.device)
        )
        angles = diffusion_step.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = nn.functional.pad(embedding, (0, 1))
        return self.mlp(embedding)


class FlowPatchEncoder(nn.Module):
    """Convert non-overlapping flow patches into GPT-2 input embeddings."""

    def __init__(self, embedding_dimension):
        super().__init__()
        if LOOKBACK % PATCH_SIZE != 0:
            raise ValueError("LOOKBACK must be divisible by PATCH_SIZE.")
        self.projection = nn.Sequential(
            nn.Linear(PATCH_SIZE, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.Dropout(DROPOUT),
        )

    def forward(self, flow):
        batch, length, channels = flow.shape
        if channels != 1:
            raise ValueError("The selected model accepts natural flow only.")
        patches = flow.reshape(batch, length // PATCH_SIZE, PATCH_SIZE)
        return self.projection(patches)


class FlowForecastBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = GPT2Model.from_pretrained(LLM_NAME)
        self.embedding_dimension = self.llm.config.n_embd
        for parameter in self.llm.parameters():
            parameter.requires_grad = False

        self.encoder = FlowPatchEncoder(self.embedding_dimension)
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dimension),
            nn.Dropout(DROPOUT),
            nn.Linear(self.embedding_dimension, HORIZON),
        )

    def forward(self, flow):
        tokens = self.encoder(flow)
        batch, token_count, _ = tokens.shape
        position_ids = torch.arange(token_count, device=flow.device)[None].expand(
            batch, -1
        )

        # Keep the frozen GPT-2 deterministic while retaining gradients with
        # respect to the trainable patch embeddings.
        self.llm.eval()
        hidden = self.llm(
            inputs_embeds=tokens,
            position_ids=position_ids,
            return_dict=True,
        ).last_hidden_state
        condition = hidden.mean(dim=1)
        return self.forecast_head(condition), condition


class ConditionalResidualDenoiser(nn.Module):
    """Denoise forecast residuals conditional on flow history and time step."""

    def __init__(self, condition_dimension):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(condition_dimension)
        input_dimension = HORIZON + 2 * condition_dimension
        self.input_layer = nn.Linear(input_dimension, DIFFUSION_HIDDEN)
        self.blocks = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DIFFUSION_HIDDEN, DIFFUSION_HIDDEN),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(DIFFUSION_HIDDEN, DIFFUSION_HIDDEN),
            nn.SiLU(),
        )
        self.output_layer = nn.Linear(DIFFUSION_HIDDEN, HORIZON)

    def forward(self, noisy_residual, condition, diffusion_step):
        time_embedding = self.time_embedding(diffusion_step)
        inputs = torch.cat((noisy_residual, condition, time_embedding), dim=1)
        hidden = self.input_layer(inputs)
        transformed = self.blocks(hidden)
        return self.output_layer(transformed + hidden)


class FlowOnlyConditionalDiffusionLLM(nn.Module):
    """Best architecture: flow-only GPT-2 with conditional residual diffusion."""

    def __init__(self):
        super().__init__()
        self.backbone = FlowForecastBackbone()
        self.denoiser = ConditionalResidualDenoiser(
            self.backbone.embedding_dimension
        )

        beta = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def diffusion_loss(self, residual, condition):
        diffusion_step = torch.randint(
            DIFFUSION_STEPS, (residual.shape[0],), device=residual.device
        )
        noise = torch.randn_like(residual)
        alpha_bar = self.alpha_bar[diffusion_step, None]
        noisy_residual = (
            alpha_bar.sqrt() * residual
            + (1.0 - alpha_bar).sqrt() * noise
        )
        predicted_noise = self.denoiser(
            noisy_residual, condition, diffusion_step
        )
        return nn.functional.mse_loss(predicted_noise, noise)

    def forward(self, flow, target):
        base_forecast, condition = self.backbone(flow)
        forecast_loss = nn.functional.mse_loss(base_forecast, target)
        residual = (target - base_forecast).detach()
        diffusion_loss = self.diffusion_loss(residual, condition)
        return forecast_loss, diffusion_loss

    @torch.inference_mode()
    def predict(self, flow, samples=DIFFUSION_SAMPLES):
        base_forecast, condition = self.backbone(flow)
        batch = flow.shape[0]

        # Process all Monte Carlo residual samples in parallel.
        condition = condition[:, None, :].expand(-1, samples, -1)
        condition = condition.reshape(batch * samples, -1)
        residual = torch.randn(
            batch * samples,
            HORIZON,
            device=flow.device,
            dtype=flow.dtype,
        )

        for step in reversed(range(DIFFUSION_STEPS)):
            diffusion_step = torch.full(
                (batch * samples,), step, device=flow.device, dtype=torch.long
            )
            predicted_noise = self.denoiser(
                residual, condition, diffusion_step
            )
            residual = (
                residual
                - self.beta[step]
                / torch.sqrt(1.0 - self.alpha_bar[step])
                * predicted_noise
            ) / torch.sqrt(self.alpha[step])
            if step > 0:
                residual = residual + self.beta[step].sqrt() * torch.randn_like(
                    residual
                )

        mean_residual = residual.reshape(batch, samples, HORIZON).mean(dim=1)
        return base_forecast + mean_residual


def train_model(model, loader):
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for flow, target in loader:
            flow = flow.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            forecast_loss, diffusion_loss = model(flow, target)
            loss = forecast_loss + DIFFUSION_WEIGHT * diffusion_loss
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item()

        history.append(running_loss / len(loader))
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch={epoch:03d} loss={history[-1]:.6f}")

    return history


@torch.inference_mode()
def evaluate(model, loader, target_mean, target_std):
    model.eval()
    predictions, targets = [], []

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for flow, target in loader:
        predictions.append(model.predict(flow.to(DEVICE)).cpu())
        targets.append(target)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start

    predictions = torch.cat(predictions).numpy() * target_std + target_mean
    targets = torch.cat(targets).numpy() * target_std + target_mean
    error = predictions - targets

    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "SMAPE": float(
            100
            * np.mean(
                2
                * np.abs(error)
                / (np.abs(targets) + np.abs(predictions) + 1e-8)
            )
        ),
        "Inference_time_s": inference_time,
        "Inference_time_per_window_s": inference_time / len(targets),
    }


def load_flow_data():
    """Load one reservoir for generalization or the original Tucurui series."""
    if not generalization_analisys:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
        return (
            pd.read_csv(
                DATA_PATH,
                sep=";",
                decimal=",",
                usecols=[FLOW_COLUMN],
            )[FLOW_COLUMN]
            .dropna()
            .to_numpy(dtype=np.float32)
        )

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in {DATA_DIR}. Run 00_download_data.py "
            "before starting this experiment."
        )

    required_columns = ["id_reservatorio", "din_instante", FLOW_COLUMN]
    reservoir_frames = []

    for csv_file in csv_files:
        monthly_data = pd.read_csv(
            csv_file,
            sep=";",
            decimal=",",
            usecols=required_columns,
            dtype={"id_reservatorio": str},
            low_memory=False,
        )
        monthly_data["id_reservatorio"] = (
            monthly_data["id_reservatorio"].str.strip()
        )
        selected_data = monthly_data.loc[
            monthly_data["id_reservatorio"] == RESERVOIR_ID,
            ["din_instante", FLOW_COLUMN],
        ]
        if not selected_data.empty:
            reservoir_frames.append(selected_data)

    if not reservoir_frames:
        raise ValueError(
            f"Reservoir {RESERVOIR_ID!r} was not found in {DATA_DIR}."
        )

    reservoir_data = pd.concat(reservoir_frames, ignore_index=True)
    reservoir_data["din_instante"] = pd.to_datetime(
        reservoir_data["din_instante"], errors="coerce"
    )
    reservoir_data[FLOW_COLUMN] = pd.to_numeric(
        reservoir_data[FLOW_COLUMN], errors="coerce"
    )
    reservoir_data = (
        reservoir_data.dropna(subset=["din_instante", FLOW_COLUMN])
        .sort_values("din_instante")
        .drop_duplicates(subset="din_instante", keep="last")
    )

    if reservoir_data.empty:
        raise ValueError(
            f"Reservoir {RESERVOIR_ID!r} has no valid flow observations."
        )

    print(
        f"Loaded {len(reservoir_data):,} observations for {RESERVOIR_ID} "
        f"from {reservoir_data['din_instante'].min().date()} to "
        f"{reservoir_data['din_instante'].max().date()}."
    )
    return reservoir_data[FLOW_COLUMN].to_numpy(dtype=np.float32)


# Load only natural flow. Precipitation is neither read nor used to filter rows.
flow = load_flow_data()

split = int(len(flow) * TRAIN_RATIO)
target_mean = float(flow[:split].mean())
target_std = float(flow[:split].std(ddof=0))
if target_std == 0:
    target_std = 1.0

# Normalization statistics are calculated exclusively from the training data.
normalized_flow = (flow - target_mean) / target_std
features = normalized_flow[:, None]

x_train, y_train = make_windows(features[:split], normalized_flow[:split])
# The pre-split lookback is historical context; every test target begins at or
# after the chronological 80/20 split.
x_test, y_test = make_windows(
    features[split - LOOKBACK :], normalized_flow[split - LOOKBACK :]
)

pin_memory = DEVICE.type == "cuda"
test_loader = DataLoader(
    TensorDataset(x_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_memory,
)

results, histories = [], []
for seed in SEEDS:
    print(f"\nRun with seed {seed}")
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
    )

    model = FlowOnlyConditionalDiffusionLLM().to(DEVICE)
    start = time.perf_counter()
    history = train_model(model, train_loader)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start
    metrics = evaluate(model, test_loader, target_mean, target_std)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    results.append(
        {
            "Architecture": "Without precipitation",
            "Seed": seed,
            "Parameters": trainable_parameters,
            "Training_time_s": training_time,
            **metrics,
        }
    )
    histories.append(history)
    print(
        f"RMSE={metrics['RMSE']:.2f} m^3/s | "
        f"MAE={metrics['MAE']:.2f} m^3/s | "
        f"SMAPE={metrics['SMAPE']:.2f}%"
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

results_df = pd.DataFrame(results)
metric_columns = [
    "RMSE",
    "MAE",
    "SMAPE",
    "Training_time_s",
    "Inference_time_s",
    "Inference_time_per_window_s",
]
summary = {
    "Architecture": "Without precipitation",
    "Runs": RUNS,
    "Parameters": int(results_df["Parameters"].iloc[0]),
}
for metric in metric_columns:
    summary[f"{metric}_mean"] = results_df[metric].mean()
    summary[f"{metric}_std"] = results_df[metric].std(ddof=1)
summary_df = pd.DataFrame([summary])

results_df.to_csv(OUTPUT_DIR / "best_model_individual_runs.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "best_model_summary.csv", index=False)
print("\nFinal flow-only conditional diffusion model")
print(summary_df.to_string(index=False))

curves = np.asarray(histories)
epochs = np.arange(1, EPOCHS + 1)
figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(epochs, curves.mean(axis=0), color="#D62728", linewidth=2.4)
axis.fill_between(
    epochs,
    curves.mean(axis=0) - curves.std(axis=0),
    curves.mean(axis=0) + curves.std(axis=0),
    color="#D62728",
    alpha=0.15,
)
axis.set_xlabel("Epoch")
axis.set_ylabel("Training loss")
axis.grid(True, alpha=0.25)
figure.tight_layout()
figure.savefig(
    OUTPUT_DIR / "best_model_learning_curve.pdf", bbox_inches="tight"
)
figure.savefig(
    OUTPUT_DIR / "best_model_learning_curve.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# ============================================================
# Display results
# ============================================================

print("\nBenchmark summary:")
print(summary_df.to_string(index=False))

print("\nLaTeX table rows:")
print(r"Model & RMSE & MAE & SMAPE (\%) & Train (s) & Test (s) \\")

for _, row in summary_df.iterrows():
    print(
        f"{row['Architecture']} & "
        f"{row['RMSE_mean']:.2E} $\\pm$ {row['RMSE_std']:.2E} & "
        f"{row['MAE_mean']:.2E} $\\pm$ {row['MAE_std']:.2E} & "
        f"{row['SMAPE_mean']:.2E} $\\pm$ {row['SMAPE_std']:.2E} & "
        f"{row['Training_time_s_mean']:.2E} & "
        f"{row['Inference_time_s_mean']:.2E} \\\\"
    )
