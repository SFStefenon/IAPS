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

gpu_number = "2"
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
elif gpu_use and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Metal Performance Shaders")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Data and experimental settings
if complete:  # Complete experiment
    EPOCHS = 100
    RUNS = 10
else:
    EPOCHS = 2
    RUNS = 2

# Reservoirs used for the generalization analysis:
# AMUHBB - Norte
# AMCLR  - Sudeste/Centro-Oeste
# CASPSO - Sul
# JEUITP - Nordeste
from pathlib import Path

if generalization_analisys:
    OUTPUT_DIR = Path("Results_05_benchmarking_generalization")
    DATA_DIR = Path("dados_hidrologicos")
    RESERVOIR_ID = "JEUITP"
    FLOW_COLUMN = "val_vazaoafluente"
else:
    OUTPUT_DIR = Path("Results_05_benchmarking")
    DATA_PATH = Path("tucurui.csv")
    FLOW_COLUMN = "Natural Flow"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_RATIO = 0.8
HORIZON = 7
SEEDS = list(range(1, RUNS + 1))
LLM_NAME = "gpt2"
DIFFUSION_SAMPLES = 10
UNCERTAINTY_SAMPLES = 100
UNCERTAINTY_WINDOWS = 256
EXPLANATION_WINDOWS = 64
INTEGRATED_GRADIENT_STEPS = 24

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
        try:
            self.llm = GPT2Model.from_pretrained(
                LLM_NAME, attn_implementation="eager"
            )
        except (TypeError, ValueError):
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

    @torch.no_grad()
    def attention_importance(self, flow):
        """Return mean attention received by each historical patch."""
        tokens = self.encoder(flow)
        batch, token_count, _ = tokens.shape
        position_ids = torch.arange(token_count, device=flow.device)[None].expand(
            batch, -1
        )
        self.llm.eval()
        output = self.llm(
            inputs_embeds=tokens,
            position_ids=position_ids,
            output_attentions=True,
            return_dict=True,
        )
        if output.attentions is None or any(
            attention is None for attention in output.attentions
        ):
            raise RuntimeError(
                "GPT-2 attention is unavailable. Use a Transformers version "
                "that supports eager attention."
            )

        # layers x batch x heads x query patches x key patches. Averaging over
        # queries is consistent with the mean pooling used by the forecast head.
        attention = torch.stack(output.attentions)
        importance = attention.mean(dim=(0, 1, 2, 3))
        return importance / importance.sum().clamp_min(1e-12)


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
    def predict(
        self, flow, samples=DIFFUSION_SAMPLES, return_samples=False
    ):
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

        residual_samples = residual.reshape(batch, samples, HORIZON)
        forecast_samples = base_forecast[:, None, :] + residual_samples
        if return_samples:
            return forecast_samples
        return forecast_samples.mean(dim=1)


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

    metrics = {
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
    return metrics, predictions, targets


@torch.inference_mode()
def predict_backbone(model, inputs, target_mean, target_std):
    """Forecast without the diffusion residual correction."""
    predictions = []
    loader = DataLoader(
        TensorDataset(inputs), batch_size=BATCH_SIZE, shuffle=False
    )
    model.eval()
    for (flow_batch,) in loader:
        forecast, _ = model.backbone(flow_batch.to(DEVICE))
        predictions.append(forecast.cpu())
    predictions = torch.cat(predictions).numpy()
    return predictions * target_std + target_mean


@torch.inference_mode()
def predict_distribution(model, inputs, target_mean, target_std):
    """Return diffusion samples without adding time to benchmark inference."""
    sample_batches = []
    loader = DataLoader(
        TensorDataset(inputs), batch_size=BATCH_SIZE, shuffle=False
    )
    model.eval()
    for (flow_batch,) in loader:
        sample_batches.append(
            model.predict(
                flow_batch.to(DEVICE),
                samples=UNCERTAINTY_SAMPLES,
                return_samples=True,
            ).cpu()
        )
    samples = torch.cat(sample_batches).numpy()
    return samples * target_std + target_mean


def temporal_importance(model, inputs):
    """Integrated-gradient importance for every lag and forecast horizon."""
    model.eval()
    flow = inputs[:EXPLANATION_WINDOWS].to(DEVICE)
    baseline = torch.zeros_like(flow)
    accumulated = torch.zeros(
        HORIZON, *flow.shape, device=flow.device, dtype=flow.dtype
    )
    alphas = torch.linspace(
        0.0, 1.0, INTEGRATED_GRADIENT_STEPS + 1, device=DEVICE
    )

    # Trapezoidal integration avoids overweighting the two endpoints.
    for index, alpha in enumerate(alphas):
        interpolated = (baseline + alpha * (flow - baseline)).detach()
        interpolated.requires_grad_(True)
        forecast, _ = model.backbone(interpolated)
        weight = 0.5 if index in (0, len(alphas) - 1) else 1.0
        for horizon_index in range(HORIZON):
            gradient = torch.autograd.grad(
                forecast[:, horizon_index].mean(),
                interpolated,
                retain_graph=horizon_index < HORIZON - 1,
            )[0]
            accumulated[horizon_index] += weight * gradient.detach()

    average_gradient = accumulated / INTEGRATED_GRADIENT_STEPS
    attribution = (flow - baseline)[None] * average_gradient
    horizon_importance = attribution.abs().mean(dim=(1, 3)).cpu().numpy()
    horizon_importance /= np.maximum(
        horizon_importance.sum(axis=1, keepdims=True), 1e-12
    )
    aggregate_importance = horizon_importance.mean(axis=0)
    aggregate_importance /= max(aggregate_importance.sum(), 1e-12)
    return aggregate_importance, horizon_importance


@torch.inference_mode()
def patch_attention(model, inputs):
    """GPT-2 attention averaged over representative chronological windows."""
    selected = inputs[:EXPLANATION_WINDOWS]
    loader = DataLoader(
        TensorDataset(selected), batch_size=BATCH_SIZE, shuffle=False
    )
    weighted_sum = None
    number_of_windows = 0
    model.eval()

    for (flow_batch,) in loader:
        batch_importance = model.backbone.attention_importance(
            flow_batch.to(DEVICE)
        ).cpu()
        batch_size = len(flow_batch)
        contribution = batch_importance * batch_size
        weighted_sum = (
            contribution if weighted_sum is None else weighted_sum + contribution
        )
        number_of_windows += batch_size

    importance = (weighted_sum / number_of_windows).numpy()
    return importance / max(importance.sum(), 1e-12)


@torch.inference_mode()
def exact_patch_shap(model, inputs, target_std, seed):
    """Exact two-group SHAP values for the complete stochastic model."""
    if LOOKBACK // PATCH_SIZE != 2:
        raise ValueError(
            "Exact patch SHAP requires the optimized two-patch configuration."
        )

    flow = inputs[:EXPLANATION_WINDOWS].to(DEVICE)
    baseline = torch.zeros_like(flow)
    older_only = baseline.clone()
    older_only[:, :PATCH_SIZE] = flow[:, :PATCH_SIZE]
    recent_only = baseline.clone()
    recent_only[:, PATCH_SIZE:] = flow[:, PATCH_SIZE:]

    # Resetting the seed gives every coalition identical diffusion noise, so
    # differences are attributable to the input patches rather than Monte Carlo
    # variation.
    def common_noise_prediction(values):
        set_seed(seed)
        return model.predict(values).cpu()

    value_none = common_noise_prediction(baseline)
    value_older = common_noise_prediction(older_only)
    value_recent = common_noise_prediction(recent_only)
    value_both = common_noise_prediction(flow)

    shap_older = 0.5 * (
        value_older - value_none + value_both - value_recent
    )
    shap_recent = 0.5 * (
        value_recent - value_none + value_both - value_older
    )
    importance = torch.stack(
        (shap_older.abs().mean(), shap_recent.abs().mean())
    )
    return importance.numpy() * target_std


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
explanation_values = None
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
    explain_this_run = seed == SEEDS[-1]
    metrics, run_predictions, run_targets = evaluate(
        model, test_loader, target_mean, target_std
    )
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

    # Explain the final run. In the complete setting RUNS=1, so this is the
    # same trained model used for the reported forecasting metrics.
    if explain_this_run:
        lag_importance, horizon_importance = temporal_importance(model, x_test)
        attention_importance = patch_attention(model, x_test)
        shap_importance = exact_patch_shap(
            model, x_test, target_std, seed + 10_000
        )
        backbone_predictions = predict_backbone(
            model, x_test, target_mean, target_std
        )
        extreme_index = int(np.argmax(run_targets.max(axis=1)))
        uncertainty_indices = np.linspace(
            0,
            len(x_test) - 1,
            min(UNCERTAINTY_WINDOWS, len(x_test)),
            dtype=int,
        )
        uncertainty_indices = np.unique(
            np.append(uncertainty_indices, extreme_index)
        )
        set_seed(seed + 20_000)
        predictive_samples = predict_distribution(
            model,
            x_test[uncertainty_indices],
            target_mean,
            target_std,
        )
        backbone_error = backbone_predictions - run_targets
        full_error = run_predictions - run_targets
        explanation_values = {
            "lag_importance": lag_importance,
            "horizon_importance": horizon_importance,
            "attention_importance": attention_importance,
            "shap_importance": shap_importance,
            "predictions": run_predictions,
            "targets": run_targets,
            "samples": predictive_samples,
            "uncertainty_indices": uncertainty_indices,
            "extreme_index": extreme_index,
            "backbone_predictions": backbone_predictions,
            "component_rmse": np.array(
                [
                    np.sqrt(np.mean(backbone_error**2)),
                    np.sqrt(np.mean(full_error**2)),
                ]
            ),
            "component_mae": np.array(
                [
                    np.mean(np.abs(backbone_error)),
                    np.mean(np.abs(full_error)),
                ]
            ),
        }

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
    summary[f"{metric}_std"] = (
        results_df[metric].std(ddof=1) if RUNS > 1 else 0.0
    )
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
plt.show()

# ============================================================
# Explainability results: compact reference-style 2 x 2 figure
# ============================================================

if explanation_values is None:
    raise RuntimeError("Explainability values were not calculated.")

attention_importance = explanation_values["attention_importance"]
shap_importance = explanation_values["shap_importance"]
lag_importance = explanation_values["lag_importance"]
component_rmse = explanation_values["component_rmse"]
component_mae = explanation_values["component_mae"]

patch_names = ["Older history", "Recent history"]
patch_positions = np.arange(len(patch_names))
patch_colors = plt.cm.turbo(np.linspace(0, 1, len(patch_names)))
lags = np.arange(-LOOKBACK, 0)
lag_colors = plt.cm.turbo(np.linspace(0, 1, LOOKBACK))
metric_colors = plt.cm.turbo(np.linspace(0, 1, 2))

figure, axes = plt.subplots(2, 2, figsize=(10, 7))

axes[0, 0].barh(
    patch_positions,
    100 * attention_importance,
    height=0.62,
    color=patch_colors,
)
axes[0, 0].set_yticks(patch_positions, patch_names)
axes[0, 0].set_xlabel("Mean GPT-2 attention (%)")
axes[0, 0].set_title("A)", loc="left")  #Mean GPT-2 attention

axes[0, 1].barh(
    patch_positions,
    shap_importance,
    height=0.62,
    color=patch_colors,
)
axes[0, 1].set_yticks(patch_positions, patch_names)
axes[0, 1].set_xlabel(r"Mean $|$SHAP$|$ (m$^3$/s)")
axes[0, 1].set_title("B)", loc="left") #  Patch SHAP importance

axes[1, 0].bar(
    lags,
    100 * lag_importance,
    width=0.9,
    color=lag_colors,
)
axes[1, 0].axvline(
    -PATCH_SIZE - 0.5,
    color="black",
    linestyle="--",
    linewidth=1.2,
)
axes[1, 0].set_xlabel("Historical lag")
axes[1, 0].set_ylabel("Integrated-gradient importance (%)")
axes[1, 0].set_title("C)", loc="left") #Temporal feature importance

models = ["Backbone only", "Full model"]
x_component = np.arange(len(models))
bar_width = 0.34
axes[1, 1].bar(
    x_component - bar_width / 2,
    component_rmse,
    bar_width,
    color=metric_colors[0],
    label="RMSE",
)
axes[1, 1].bar(
    x_component + bar_width / 2,
    component_mae,
    bar_width,
    color=metric_colors[1],
    label="MAE",
)
axes[1, 1].set_xticks(x_component, models)
axes[1, 1].set_ylabel(r"Error (m$^3$/s)")
axes[1, 1].set_title("D)", loc="left") # Diffusion component analysis
axes[1, 1].legend(fontsize=7)

for axis in axes.ravel():
    axis.set_axisbelow(True)
    axis.grid(True, alpha=0.25)

figure.tight_layout()
figure.savefig(
    OUTPUT_DIR / "best_model_explainability.pdf", bbox_inches="tight"
)
plt.show()

# ============================================================
# Extended explanations: when, where, and why diffusion helps
# ============================================================

horizon_importance = explanation_values["horizon_importance"]
full_predictions = explanation_values["predictions"]
targets = explanation_values["targets"]
backbone_predictions = explanation_values["backbone_predictions"]
predictive_samples = explanation_values["samples"]
uncertainty_indices = explanation_values["uncertainty_indices"]
extreme_index = explanation_values["extreme_index"]

backbone_horizon_rmse = np.sqrt(
    np.mean((backbone_predictions - targets) ** 2, axis=0)
)
full_horizon_rmse = np.sqrt(
    np.mean((full_predictions - targets) ** 2, axis=0)
)
diffusion_gain = 100 * (
    backbone_horizon_rmse - full_horizon_rmse
) / np.maximum(backbone_horizon_rmse, 1e-12)

# Regime thresholds are calculated from training data only.
q25, q75, q90 = np.quantile(flow[:split], [0.25, 0.75, 0.90])
target_flat = targets.ravel()
backbone_flat = backbone_predictions.ravel()
full_flat = full_predictions.ravel()
regime_names = ["Low", "Normal", "High", "Extreme"]
regime_masks = [
    target_flat <= q25,
    (target_flat > q25) & (target_flat <= q75),
    (target_flat > q75) & (target_flat <= q90),
    target_flat > q90,
]
backbone_regime_rmse, full_regime_rmse = [], []
for mask in regime_masks:
    if not np.any(mask):
        backbone_regime_rmse.append(np.nan)
        full_regime_rmse.append(np.nan)
        continue
    backbone_regime_rmse.append(
        np.sqrt(np.mean((backbone_flat[mask] - target_flat[mask]) ** 2))
    )
    full_regime_rmse.append(
        np.sqrt(np.mean((full_flat[mask] - target_flat[mask]) ** 2))
    )
backbone_regime_rmse = np.asarray(backbone_regime_rmse)
full_regime_rmse = np.asarray(full_regime_rmse)

uncertainty_targets = targets[uncertainty_indices]
lower_interval = np.quantile(predictive_samples, 0.05, axis=1)
upper_interval = np.quantile(predictive_samples, 0.95, axis=1)
uncertainty_mean = predictive_samples.mean(axis=1)
interval_coverage = 100 * np.mean(
    (uncertainty_targets >= lower_interval)
    & (uncertainty_targets <= upper_interval)
)
mean_interval_width = np.mean(upper_interval - lower_interval)
extreme_position = int(np.flatnonzero(uncertainty_indices == extreme_index)[0])









figure, axes = plt.subplots(2, 2, figsize=(10, 7))

image = axes[0, 0].imshow(
    100 * horizon_importance,
    aspect="auto",
    origin="lower",
    cmap="turbo",
    extent=[-LOOKBACK - 0.5, -0.5, 0.5, HORIZON + 0.5],
)
axes[0, 0].set_xlabel("Historical lag")
axes[0, 0].set_ylabel("Forecast horizon")
axes[0, 0].set_yticks(np.arange(1, HORIZON + 1))
axes[0, 0].set_title("A)", loc="left") # Lag--horizon feature importance
colorbar = figure.colorbar(image, ax=axes[0, 0], pad=0.02)
colorbar.set_label("Importance (%)")

horizons = np.arange(1, HORIZON + 1)
bar_width = 0.34
axes[0, 1].bar(
    horizons - bar_width / 2,
    backbone_horizon_rmse,
    bar_width,
    color=metric_colors[0],
    label="Backbone",
)
axes[0, 1].bar(
    horizons + bar_width / 2,
    full_horizon_rmse,
    bar_width,
    color=metric_colors[1],
    label="Full model",
)
gain_axis = axes[0, 1].twinx()
gain_axis.grid(False)
gain_axis.plot(
    horizons,
    diffusion_gain,
    color="black",
    marker="o",
    linewidth=1.4,
    label="RMSE gain",
)
gain_axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[0, 1].set_xticks(horizons)
axes[0, 1].set_xlabel("Forecast horizon")
axes[0, 1].set_ylabel(r"RMSE (m$^3$/s)")
gain_axis.set_ylabel("Diffusion gain (%)")
axes[0, 1].set_title("B)", loc="left") # Diffusion improvement by horizon
handles_1, labels_1 = axes[0, 1].get_legend_handles_labels()
handles_2, labels_2 = gain_axis.get_legend_handles_labels()
axes[0, 1].legend(
    handles_1 + handles_2,
    labels_1 + labels_2,
    fontsize=7,
)

regime_positions = np.arange(len(regime_names))
axes[1, 0].bar(
    regime_positions - bar_width / 2,
    backbone_regime_rmse,
    bar_width,
    color=metric_colors[0],
    label="Backbone",
)
axes[1, 0].bar(
    regime_positions + bar_width / 2,
    full_regime_rmse,
    bar_width,
    color=metric_colors[1],
    label="Full model",
)
axes[1, 0].set_xticks(regime_positions, regime_names)
axes[1, 0].set_ylabel(r"RMSE (m$^3$/s)")
axes[1, 0].set_title("C)", loc="left") #Performance by flow regime
axes[1, 0].legend(fontsize=7)

example_horizons = np.arange(1, HORIZON + 1)
axes[1, 1].fill_between(
    example_horizons,
    lower_interval[extreme_position],
    upper_interval[extreme_position],
    color=metric_colors[1],
    alpha=0.18,
    label="90% interval",
)
axes[1, 1].plot(
    example_horizons,
    targets[extreme_index],
    color="black",
    marker="o",
    linewidth=1.8,
    label="Observed",
)
axes[1, 1].plot(
    example_horizons,
    backbone_predictions[extreme_index],
    color=metric_colors[0],
    marker="s",
    linewidth=1.4,
    label="Backbone",
)
axes[1, 1].plot(
    example_horizons,
    uncertainty_mean[extreme_position],
    color=metric_colors[1],
    marker="^",
    linewidth=1.6,
    label="Full model",
)
axes[1, 1].set_xticks(example_horizons)
axes[1, 1].set_xlabel("Forecast horizon")
axes[1, 1].set_ylabel(r"Natural flow (m$^3$/s)")
axes[1, 1].set_title("D)", loc="left") # Extreme-flow prediction uncertainty
axes[1, 1].legend(fontsize=7)
axes[1, 1].text(
    0.02,
    0.96,
    f"Coverage = {interval_coverage:.1f}%\n"
    f"Mean width = {mean_interval_width:.1f} m$^3$/s",
    transform=axes[1, 1].transAxes,
    ha="left",
    va="top",
    fontsize=7,
)

for axis in axes.ravel():
    axis.grid(True, alpha=0.25)

figure.tight_layout()
figure.savefig(
    OUTPUT_DIR / "best_model_explainability_extended.pdf",
    bbox_inches="tight",
)
plt.show()

print(
    f"\n90% interval coverage={interval_coverage:.2f}% | "
    f"mean width={mean_interval_width:.2f} m^3/s"
)

# ============================================================
# Extract explainability values
# ============================================================

if explanation_values is None:
    raise RuntimeError("Explainability values were not calculated.")

attention_importance = explanation_values["attention_importance"]
shap_importance = explanation_values["shap_importance"]
lag_importance = explanation_values["lag_importance"]
component_rmse = explanation_values["component_rmse"]
component_mae = explanation_values["component_mae"]
horizon_importance = explanation_values["horizon_importance"]
full_predictions = explanation_values["predictions"]
targets = explanation_values["targets"]
backbone_predictions = explanation_values["backbone_predictions"]
predictive_samples = explanation_values["samples"]
uncertainty_indices = explanation_values["uncertainty_indices"]
extreme_index = explanation_values["extreme_index"]

patch_names = ["Older history", "Recent history"]
patch_positions = np.arange(len(patch_names))
patch_colors = plt.cm.turbo(np.linspace(0, 1, len(patch_names)))
metric_colors = plt.cm.turbo(np.linspace(0, 1, 2))
lags = np.arange(-LOOKBACK, 0)
lag_colors = plt.cm.turbo(np.linspace(0, 1, LOOKBACK))


# ============================================================
# Learning curve
# ============================================================

curves = np.asarray(histories)
epochs = np.arange(1, EPOCHS + 1)

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

axis.plot(
    epochs,
    curves.mean(axis=0),
    color="#D62728",
    linewidth=2.4,
)

axis.fill_between(
    epochs,
    curves.mean(axis=0) - curves.std(axis=0),
    curves.mean(axis=0) + curves.std(axis=0),
    color="#D62728",
    alpha=0.15,
)

axis.set_xlabel("Epoch")
axis.set_ylabel("Training loss")
axis.set_axisbelow(True)
axis.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "best_model_learning_curve.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Mean GPT-2 attention
# ============================================================

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

axis.barh(
    patch_positions,
    100 * attention_importance,
    height=0.62,
    color=patch_colors,
)

axis.set_yticks(
    patch_positions,
    patch_names,
)

axis.set_xlabel("Mean GPT-2 attention (%)")
axis.set_axisbelow(True)
axis.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "explainability_attention.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Patch SHAP importance
# ============================================================

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

axis.barh(
    patch_positions,
    shap_importance,
    height=0.62,
    color=patch_colors,
)

axis.set_yticks(
    patch_positions,
    patch_names,
)

axis.set_xlabel(r"Mean $|$SHAP$|$ (m$^3$/s)")
axis.set_axisbelow(True)
axis.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "explainability_patch_shap.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Temporal feature importance
# ============================================================

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

axis.bar(
    lags,
    100 * lag_importance,
    width=0.9,
    color=lag_colors,
)

axis.axvline(
    -PATCH_SIZE - 0.5,
    color="black",
    linestyle="--",
    linewidth=1.2,
)

axis.set_xlabel("Historical lag")
axis.set_ylabel("Integrated-gradient importance (%)")
axis.set_axisbelow(True)
axis.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "explainability_temporal_importance.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Diffusion component analysis
# ============================================================

models = ["Backbone only", "Full model"]
x_component = np.arange(len(models))
bar_width = 0.34

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

axis.bar(
    x_component - bar_width / 2,
    component_rmse,
    bar_width,
    color=metric_colors[0],
    label="RMSE",
)

axis.bar(
    x_component + bar_width / 2,
    component_mae,
    bar_width,
    color=metric_colors[1],
    label="MAE",
)

axis.set_xticks(
    x_component,
    models,
)

axis.set_ylabel(r"Error (m$^3$/s)")
axis.legend(frameon=True)
axis.set_axisbelow(True)
axis.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "explainability_diffusion_component.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Lag-horizon feature importance
# ============================================================

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

image = axis.imshow(
    100 * horizon_importance,
    aspect="auto",
    origin="lower",
    cmap="turbo",
    extent=[
        -LOOKBACK - 0.5,
        -0.5,
        0.5,
        HORIZON + 0.5,
    ],
)

axis.set_xlabel("Historical lag")
axis.set_ylabel("Forecast horizon")
axis.set_yticks(np.arange(1, HORIZON + 1))

colorbar = figure.colorbar(
    image,
    ax=axis,
    pad=0.02,
)

colorbar.set_label("Importance (%)")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "explainability_lag_horizon.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Diffusion improvement by horizon
# ============================================================

backbone_horizon_rmse = np.sqrt(
    np.mean(
        (backbone_predictions - targets) ** 2,
        axis=0,
    )
)

full_horizon_rmse = np.sqrt(
    np.mean(
        (full_predictions - targets) ** 2,
        axis=0,
    )
)

diffusion_gain = (
    100
    * (backbone_horizon_rmse - full_horizon_rmse)
    / np.maximum(backbone_horizon_rmse, 1e-12)
)

horizons = np.arange(1, HORIZON + 1)
bar_width = 0.34

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

axis.bar(
    horizons - bar_width / 2,
    backbone_horizon_rmse,
    bar_width,
    color=metric_colors[0],
    label="Backbone",
    zorder=2,
)

axis.bar(
    horizons + bar_width / 2,
    full_horizon_rmse,
    bar_width,
    color=metric_colors[1],
    label="Full model",
    zorder=2,
)

gain_axis = axis.twinx()
gain_axis.grid(False)

gain_axis.plot(
    horizons,
    diffusion_gain,
    color="black",
    marker="o",
    linewidth=1.4,
    label="RMSE gain",
    zorder=3,
)

gain_axis.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=0.8,
)

axis.set_xticks(horizons)
axis.set_xlabel("Forecast horizon")
axis.set_ylabel(r"RMSE (m$^3$/s)")
gain_axis.set_ylabel("Diffusion gain (%)")

handles_1, labels_1 = axis.get_legend_handles_labels()
handles_2, labels_2 = gain_axis.get_legend_handles_labels()

axis.legend(
    handles_1 + handles_2,
    labels_1 + labels_2,
    fontsize=8,
    frameon=True,
)

axis.set_axisbelow(True)
axis.grid(True, alpha=0.25)
gain_axis.grid(False)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "explainability_diffusion_horizon.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Performance by flow regime
# ============================================================

q25, q75, q90 = np.quantile(
    flow[:split],
    [0.25, 0.75, 0.90],
)

target_flat = targets.ravel()
backbone_flat = backbone_predictions.ravel()
full_flat = full_predictions.ravel()

regime_names = ["Low", "Normal", "High", "Extreme"]

regime_masks = [
    target_flat <= q25,
    (target_flat > q25) & (target_flat <= q75),
    (target_flat > q75) & (target_flat <= q90),
    target_flat > q90,
]

backbone_regime_rmse = []
full_regime_rmse = []

for mask in regime_masks:
    if not np.any(mask):
        backbone_regime_rmse.append(np.nan)
        full_regime_rmse.append(np.nan)
        continue

    backbone_regime_rmse.append(
        np.sqrt(
            np.mean(
                (backbone_flat[mask] - target_flat[mask]) ** 2
            )
        )
    )

    full_regime_rmse.append(
        np.sqrt(
            np.mean(
                (full_flat[mask] - target_flat[mask]) ** 2
            )
        )
    )

backbone_regime_rmse = np.asarray(backbone_regime_rmse)
full_regime_rmse = np.asarray(full_regime_rmse)

regime_positions = np.arange(len(regime_names))
bar_width = 0.34

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

axis.bar(
    regime_positions - bar_width / 2,
    backbone_regime_rmse,
    bar_width,
    color=metric_colors[0],
    label="Backbone",
)

axis.bar(
    regime_positions + bar_width / 2,
    full_regime_rmse,
    bar_width,
    color=metric_colors[1],
    label="Full model",
)

axis.set_xticks(
    regime_positions,
    regime_names,
)

axis.set_ylabel(r"RMSE (m$^3$/s)")
axis.legend(frameon=True)
axis.set_axisbelow(True)
axis.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "explainability_flow_regime.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Extreme-flow prediction uncertainty
# ============================================================

uncertainty_targets = targets[uncertainty_indices]

lower_interval = np.quantile(
    predictive_samples,
    0.05,
    axis=1,
)

upper_interval = np.quantile(
    predictive_samples,
    0.95,
    axis=1,
)

uncertainty_mean = predictive_samples.mean(axis=1)

interval_coverage = 100 * np.mean(
    (uncertainty_targets >= lower_interval)
    & (uncertainty_targets <= upper_interval)
)

mean_interval_width = np.mean(
    upper_interval - lower_interval
)

extreme_position = int(
    np.flatnonzero(
        uncertainty_indices == extreme_index
    )[0]
)

example_horizons = np.arange(1, HORIZON + 1)

figure = plt.figure(figsize=(7, 4))
axis = plt.gca()

axis.fill_between(
    example_horizons,
    lower_interval[extreme_position],
    upper_interval[extreme_position],
    color=metric_colors[1],
    alpha=0.18,
    label="90% interval",
)

axis.plot(
    example_horizons,
    targets[extreme_index],
    color="black",
    marker="o",
    linewidth=1.8,
    label="Observed",
)

axis.plot(
    example_horizons,
    backbone_predictions[extreme_index],
    color=metric_colors[0],
    marker="s",
    linewidth=1.4,
    label="Backbone",
)

axis.plot(
    example_horizons,
    uncertainty_mean[extreme_position],
    color=metric_colors[1],
    marker="^",
    linewidth=1.6,
    label="Full model",
)

axis.set_xticks(example_horizons)
axis.set_xlabel("Forecast horizon")
axis.set_ylabel(r"Natural flow (m$^3$/s)")

axis.legend(
    fontsize=8,
    frameon=True,
)

axis.text(
    0.02,
    0.96,
    (
        f"Coverage = {interval_coverage:.1f}%\n"
        f"Mean width = {mean_interval_width:.1f} m$^3$/s"
    ),
    transform=axis.transAxes,
    ha="left",
    va="top",
    fontsize=8,
)

axis.set_axisbelow(True)
axis.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "explainability_uncertainty.pdf",
    bbox_inches="tight",
)
plt.show()


# ============================================================
# Uncertainty summary
# ============================================================

print(
    f"\n90% interval coverage={interval_coverage:.2f}% | "
    f"mean width={mean_interval_width:.2f} m^3/s"
)


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
