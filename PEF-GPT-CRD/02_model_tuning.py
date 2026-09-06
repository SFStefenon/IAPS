import gc
import json
import os
import random
import time
import warnings

GPU_USE = True
GPU_NUMBER = "3"

if GPU_USE:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUMBER

os.environ.update(
    {
        "USE_TF": "0",
        "USE_FLAX": "0",
        "USE_TORCH": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
)
warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from optuna.importance import get_param_importances
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model


DEVICE = torch.device(
    "cuda"
    if GPU_USE and torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if DEVICE.type == "cuda":
    print(f"GPU {GPU_NUMBER}: {torch.cuda.get_device_name(0)}")
else:
    print(f"Device: {DEVICE}")

DATA_FILE = "tucurui.csv"
FLOW_COLUMN = "Natural Flow"
MODEL_NAME = "Diffusion-LLM without precipitation"
LLM_NAME = "gpt2"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.20
HORIZON = 7

N_TRIALS = 100
TUNING_SEED = 0
MAX_TUNING_EPOCHS = 50
VALIDATION_INTERVAL = 50
FINAL_EPOCHS = 100
FINAL_SEEDS = list(range(1, 11))

RESOURCE_LEVELS = {5, 15, 45, MAX_TUNING_EPOCHS}
STUDY_NAME = "flow_only_diffusion_llm_mtpe_hyperband"
STORAGE = f"sqlite:///{STUDY_NAME}.db"

TRIALS_FILE = "hypertuning_trials.csv"
BEST_PARAMETERS_FILE = "hypertuning_best_hyperparameters.json"
FINAL_RUNS_FILE = "hypertuning_without_precipitation_individual_runs.csv"
FINAL_SUMMARY_FILE = "hypertuning_without_precipitation_summary.csv"
HYPERTUNING_PLOT_PDF = "hypertuning_analysis.pdf"
HYPERTUNING_PLOT_PNG = "hypertuning_analysis.png"
PARALLEL_PLOT_PDF = "hypertuning_parallel_coordinates.pdf"
PARALLEL_PLOT_PNG = "hypertuning_parallel_coordinates.png"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_windows(series, target_start, target_end, lookback):
    """Create windows whose forecast targets stay inside one chronological split."""
    last_start = target_end - HORIZON
    if target_start < lookback or last_start < target_start:
        raise ValueError("The selected split is too short for this configuration.")

    starts = range(target_start, last_start + 1)
    x = np.stack([series[i - lookback : i] for i in starts])[..., None]
    y = np.stack([series[i : i + HORIZON] for i in starts])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_flow():
    # Precipitation is neither loaded nor used to filter the observations.
    data = pd.read_csv(
        DATA_FILE,
        sep=";",
        decimal=",",
        usecols=[FLOW_COLUMN],
    )
    flow = (
        pd.to_numeric(data[FLOW_COLUMN], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(np.float32)
        .to_numpy()
    )

    n = len(flow)
    train_end = int(n * TRAIN_RATIO)
    validation_end = train_end + int(n * VALIDATION_RATIO)

    if validation_end >= n or not np.isclose(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO, 1.0
    ):
        raise ValueError("Invalid train/validation/test split.")

    flow_mean = float(flow[:train_end].mean())
    flow_std = float(flow[:train_end].std()) or 1.0
    normalized = (flow - flow_mean) / flow_std
    if not np.isfinite(normalized).all():
        raise ValueError("The normalized natural-flow series contains non-finite values.")
    return normalized, train_end, validation_end, flow_mean, flow_std


def make_datasets(series, train_end, validation_end, lookback):
    train = make_windows(series, lookback, train_end, lookback)
    validation = make_windows(series, train_end, validation_end, lookback)
    test = make_windows(series, validation_end, len(series), lookback)
    return train, validation, test


def make_loader(dataset, batch_size, shuffle=False, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        TensorDataset(*dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=DEVICE.type == "cuda",
    )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, t):
        half = self.dimension // 2
        scale = np.log(10000) / max(half - 1, 1)
        frequency = torch.exp(-scale * torch.arange(half, device=t.device))
        angle = t.float()[:, None] * frequency[None]
        embedding = torch.cat((angle.sin(), angle.cos()), dim=1)

        if embedding.shape[1] < self.dimension:
            embedding = nn.functional.pad(embedding, (0, 1))

        return self.mlp(embedding)


class FlowPatchEncoder(nn.Module):
    """Encode non-overlapping patches containing only historical natural flow."""

    def __init__(self, lookback, patch_size, dimension, dropout):
        super().__init__()
        if lookback % patch_size != 0:
            raise ValueError("lookback must be divisible by patch_size.")

        self.lookback = lookback
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Linear(patch_size, dimension),
            nn.LayerNorm(dimension),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        batch, length, channels = x.shape
        if channels != 1:
            raise ValueError("This model accepts only the natural-flow channel.")
        if length != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, received {length}.")

        patches = x.reshape(batch, length // self.patch_size, self.patch_size)
        return self.projection(patches)


class FrozenGPT2Backbone(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.network = GPT2Model.from_pretrained(LLM_NAME)
        self.dimension = self.network.config.n_embd

        for parameter in self.network.parameters():
            parameter.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(self.dimension),
            nn.Dropout(dropout),
            nn.Linear(self.dimension, HORIZON),
        )

    def forward(self, tokens):
        self.network.eval()
        batch, length, _ = tokens.shape
        positions = torch.arange(length, device=tokens.device)[None].expand(
            batch, -1
        )
        hidden = self.network(
            inputs_embeds=tokens,
            position_ids=positions,
            return_dict=True,
        ).last_hidden_state
        condition = hidden.mean(dim=1)
        return self.head(condition), condition


class ConditionalResidualDenoiser(nn.Module):
    def __init__(self, condition_dimension, hidden_dimension, dropout):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(condition_dimension)
        input_dimension = HORIZON + 2 * condition_dimension

        self.input_layer = nn.Linear(input_dimension, hidden_dimension)
        self.hidden_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
        )
        self.output_layer = nn.Linear(hidden_dimension, HORIZON)

    def forward(self, noisy_residual, condition, t):
        inputs = torch.cat(
            (noisy_residual, condition, self.time_embedding(t)), dim=1
        )
        hidden = self.input_layer(inputs)
        transformed = self.hidden_layers(hidden)
        return self.output_layer(transformed + hidden)


class FlowDiffusionLLM(nn.Module):
    """Diffusion-LLM forecaster using natural flow as its only input variable."""

    def __init__(self, config):
        super().__init__()
        self.diffusion_steps = config["diffusion_steps"]
        self.backbone = FrozenGPT2Backbone(config["dropout"])
        self.encoder = FlowPatchEncoder(
            config["lookback"],
            config["patch_size"],
            self.backbone.dimension,
            config["dropout"],
        )
        self.denoiser = ConditionalResidualDenoiser(
            self.backbone.dimension,
            config["diffusion_hidden"],
            config["dropout"],
        )

        beta = torch.linspace(1e-4, 0.02, self.diffusion_steps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)

    def encode(self, x):
        return self.backbone(self.encoder(x))

    def diffusion_loss(self, residual, condition):
        t = torch.randint(
            self.diffusion_steps,
            (residual.shape[0],),
            device=residual.device,
        )
        noise = torch.randn_like(residual)
        alpha_bar_t = self.alpha_bar[t, None]
        noisy_residual = (
            alpha_bar_t.sqrt() * residual
            + (1 - alpha_bar_t).sqrt() * noise
        )
        predicted_noise = self.denoiser(noisy_residual, condition, t)
        return nn.functional.mse_loss(predicted_noise, noise)

    def forward(self, x, y):
        base_forecast, condition = self.encode(x)
        forecast_loss = nn.functional.mse_loss(base_forecast, y)
        residual = (y - base_forecast).detach()
        diffusion_loss = self.diffusion_loss(residual, condition)
        return forecast_loss, diffusion_loss

    @torch.inference_mode()
    def predict(self, x):
        # Diffusion is an auxiliary training objective. Inference uses only the
        # deterministic forecasting head, without reverse-diffusion sampling.
        base_forecast, _ = self.encode(x)
        return base_forecast


def make_optimizer(model, config):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    return optimizer, parameters


def train_one_epoch(model, loader, optimizer, parameters, config):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        forecast_loss, diffusion_loss = model(x, y)
        loss = forecast_loss + config["diffusion_weight"] * diffusion_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("The training loss became non-finite.")
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, config["grad_clip"])
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, flow_mean, flow_std):
    model.eval()
    predictions, targets = [], []
    start = time.perf_counter()

    for x, y in loader:
        batch_predictions = model.predict(x.to(DEVICE)).cpu()
        if not torch.isfinite(batch_predictions).all():
            raise FloatingPointError("The model produced non-finite predictions.")
        predictions.append(batch_predictions)
        targets.append(y)

    inference_time = time.perf_counter() - start
    predictions = torch.cat(predictions).numpy().ravel() * flow_std + flow_mean
    targets = torch.cat(targets).numpy().ravel() * flow_std + flow_mean
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
        "Inference_time_per_window_s": inference_time / len(loader.dataset),
    }


def suggest_configuration(trial):
    config = {
        "lookback": trial.suggest_categorical(
            "lookback", [64, 128, 256, 512]
        ),
        "patch_size": trial.suggest_categorical(
            "patch_size", [4, 8, 16, 32]
        ),
        "diffusion_steps": trial.suggest_categorical(
            "diffusion_steps", [25, 50, 100, 200]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "diffusion_hidden": trial.suggest_categorical(
            "diffusion_hidden", [64, 128, 256, 512]
        ),
        "diffusion_weight": trial.suggest_float(
            "diffusion_weight", 1e-4, 0.5, log=True
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [4, 16, 32, 64]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        ),
        "grad_clip": trial.suggest_categorical(
            "grad_clip", [0.5, 1.0, 2.0, 4.0]
        ),
    }

    if config["lookback"] % config["patch_size"] != 0:
        raise optuna.TrialPruned("lookback must be divisible by patch_size")

    return config


def make_objective(series, train_end, validation_end, flow_mean, flow_std):
    def objective(trial):
        config = suggest_configuration(trial)
        set_seed(TUNING_SEED)
        train = make_windows(
            series, config["lookback"], train_end, config["lookback"]
        )
        validation = make_windows(
            series, train_end, validation_end, config["lookback"]
        )
        train_loader = make_loader(
            train,
            config["batch_size"],
            shuffle=True,
            seed=TUNING_SEED,
        )
        validation_loader = make_loader(validation, config["batch_size"])
        model = None

        try:
            model = FlowDiffusionLLM(config).to(DEVICE)
            optimizer, parameters = make_optimizer(model, config)
            trial.set_user_attr(
                "trainable_parameters",
                sum(p.numel() for p in parameters),
            )
            start = time.perf_counter()

            for epoch in range(MAX_TUNING_EPOCHS):
                train_one_epoch(model, train_loader, optimizer, parameters, config)
                resource = epoch + 1
                if resource not in RESOURCE_LEVELS:
                    continue

                validation_rmse = evaluate(
                    model,
                    validation_loader,
                    flow_mean,
                    flow_std,
                )["RMSE"]
                if not np.isfinite(validation_rmse):
                    raise optuna.TrialPruned("Non-finite validation RMSE")
                trial.report(validation_rmse, step=resource)

                if trial.should_prune():
                    trial.set_user_attr("epochs_completed", resource)
                    trial.set_user_attr(
                        "training_time_s", time.perf_counter() - start
                    )
                    raise optuna.TrialPruned()

            trial.set_user_attr("epochs_completed", MAX_TUNING_EPOCHS)
            trial.set_user_attr(
                "training_time_s", time.perf_counter() - start
            )
            return validation_rmse

        except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
            raise optuna.TrialPruned(str(error)) from error
        finally:
            if model is not None:
                del model
            clear_memory()

    return objective


def last_finite_value(trial):
    if trial.value is not None and np.isfinite(trial.value):
        return float(trial.value)

    values = [
        float(value)
        for _, value in sorted(trial.intermediate_values.items())
        if np.isfinite(value)
    ]
    return values[-1] if values else None


def format_parameter(name):
    return name.replace("_", " ").capitalize()


def style_axis(axis):
    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    axis.grid(True, color="#D9E0EA", alpha=0.65, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AAB4C3")
    axis.spines["bottom"].set_color("#AAB4C3")
    axis.tick_params(colors="#344054", labelsize=8.5)
    axis.xaxis.label.set_color("#1D2939")
    axis.yaxis.label.set_color("#1D2939")


def encode_parameter(values, parameter, jitter=False):
    numeric = all(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        for value in values
    )
    categories = sorted(set(values), key=lambda value: str(value))
    continuous = numeric and len(categories) > 5

    if continuous:
        encoded = np.asarray(values, dtype=float)
        return encoded, None, parameter in {
            "learning_rate",
            "weight_decay",
            "diffusion_weight",
        }

    mapping = {value: position for position, value in enumerate(categories)}
    encoded = np.asarray([mapping[value] for value in values], dtype=float)
    if jitter and len(values) > len(categories):
        rng = np.random.default_rng(TUNING_SEED + len(parameter))
        encoded += rng.normal(0, 0.045, len(encoded))
    return encoded, categories, False


def plot_parallel_coordinates(completed, importances):
    if len(completed) < 3:
        return

    common = set(completed[0].params)
    for trial in completed[1:]:
        common &= set(trial.params)

    ranked = [name for name in importances if name in common]
    ranked.extend(sorted(common - set(ranked)))
    parameters = ranked[:6]
    if len(parameters) < 2:
        return

    selected = sorted(completed, key=last_finite_value)[: min(25, len(completed))]
    rmse = np.asarray([last_finite_value(trial) for trial in selected])
    norm = Normalize(vmin=rmse.min(), vmax=rmse.max() + 1e-12)
    cmap = plt.cm.turbo_r
    positions = np.arange(len(parameters))
    encoded = []
    annotations = []

    for parameter in parameters:
        raw = [trial.params[parameter] for trial in selected]
        numeric = all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            for value in raw
        )
        transformed = np.log10(raw) if numeric and parameter in {
            "learning_rate",
            "weight_decay",
            "diffusion_weight",
        } else np.asarray(raw) if numeric else None

        if numeric:
            transformed = np.asarray(transformed, dtype=float)
            low, high = transformed.min(), transformed.max()
            scaled = np.zeros_like(transformed) if high == low else (
                transformed - low
            ) / (high - low)
            raw_low, raw_high = min(raw), max(raw)
            annotations.append([(0.0, f"{raw_low:.2g}"), (1.0, f"{raw_high:.2g}")])
        else:
            categories = sorted(set(raw), key=str)
            mapping = {value: index for index, value in enumerate(categories)}
            denominator = max(len(categories) - 1, 1)
            scaled = np.asarray([mapping[value] / denominator for value in raw])
            annotations.append(
                [(index / denominator, str(value)) for index, value in enumerate(categories)]
            )
        encoded.append(scaled)

    encoded = np.asarray(encoded).T
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, axis = plt.subplots(figsize=(12.5, 5.4), facecolor="white")
        axis.set_facecolor("white")

        for index in range(len(parameters)):
            if index % 2 == 0:
                axis.axvspan(index - 0.5, index + 0.5, color="#F5F8FC", zorder=0)
            axis.axvline(index, color="#B9C4D2", linewidth=1.0, zorder=1)

        order = np.argsort(rmse)[::-1]
        for index in order:
            axis.plot(
                positions,
                encoded[index],
                color=cmap(norm(rmse[index])),
                linewidth=1.25,
                alpha=0.62,
                zorder=2,
            )

        best_index = int(np.argmin(rmse))
        axis.plot(
            positions,
            encoded[best_index],
            color="r",
            linewidth=3.0,
            linestyle="--",
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            zorder=4,
            label=f"Best trial ({rmse[best_index]:.1f} m³/s)",
        )

        for position, labels in zip(positions, annotations):
            for y_value, label in labels:
                axis.text(
                    position + 0.025,
                    y_value,
                    label,
                    fontsize=6.5,
                    color="#667085",
                    va="center",
                    clip_on=False,
                )

        axis.set(
            xlim=(-0.15, len(parameters) - 0.55),
            ylim=(-0.08, 1.08),
            xticks=positions,
            xticklabels=[format_parameter(name) for name in parameters],
        )
        axis.tick_params(axis="x", labelsize=9, colors="#1D2939", pad=10)
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title("Parallel coordinates of the best trials", loc="left", fontweight="normal")
        axis.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=8)

        colorbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=axis,
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label(r"Validation RMSE ($\mathrm{m^3/s}$)")
        colorbar.outline.set_visible(False)
        fig.tight_layout()
        fig.savefig(PARALLEL_PLOT_PDF, bbox_inches="tight")
        #fig.savefig(PARALLEL_PLOT_PNG, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_hypertuning_analysis(study):
    completed = sorted(
        [
            trial
            for trial in study.trials
            if trial.state == TrialState.COMPLETE
            and last_finite_value(trial) is not None
        ],
        key=lambda trial: trial.number,
    )
    pruned = [
        trial
        for trial in study.trials
        if trial.state == TrialState.PRUNED and last_finite_value(trial) is not None
    ]
    try:
        importances = get_param_importances(study) if len(completed) >= 2 else {}
    except (RuntimeError, ValueError, ZeroDivisionError):
        importances = {}

    navy, blue, orange, red = "#14213D", "#168AAD", "#F77F00", "#D62828"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, ax = plt.subplots(2, 2, figsize=(13.2, 9.2), facecolor="white")
        fig.subplots_adjust(wspace=0.28, hspace=0.32)
        for axis in ax.ravel():
            style_axis(axis)

        # A) Optimization history with the best-so-far envelope.
        if completed:
            numbers = np.asarray([trial.number for trial in completed])
            values = np.asarray([last_finite_value(trial) for trial in completed])
            norm = Normalize(vmin=values.min(), vmax=values.max() + 1e-12)
            ax[0, 0].scatter(
                numbers,
                values,
                c=values,
                norm=norm,
                cmap="turbo_r",
                s=62,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
            running_best = np.minimum.accumulate(values)
            ax[0, 0].step(
                numbers,
                running_best,
                where="post",
                linestyle="--",
                color=red,
                linewidth=2.4,
                label="Best so far",
                zorder=3,
            )
            best_index = int(np.argmin(values))
            best_x, best_y = numbers[best_index], values[best_index]
            ax[0, 0].scatter(
                [best_x],
                [best_y],
                marker="*",
                s=230,
                color="#FFD166",
                edgecolor=navy,
                linewidth=1.1,
                zorder=6,
            )
            ax[0, 0].annotate(
                f"Best: {best_y:.1f}",
                xy=(best_x, best_y),
                xytext=(10, 18),
                textcoords="offset points",
                fontsize=8,
                fontweight="normal",
                color=navy,
                arrowprops={"arrowstyle": "-", "color": navy, "lw": 0.9},
            )

        if pruned:
            ax[0, 0].scatter(
                [trial.number for trial in pruned],
                [last_finite_value(trial) for trial in pruned],
                color=orange,
                marker="x",
                s=42,
                linewidth=1.1,
                alpha=0.75,
                label="Pruned",
                zorder=2,
            )

        ax[0, 0].set(xlabel="Trial", ylabel=r"Validation RMSE ($\mathrm{m^3/s}$)")
        ax[0, 0].set_title("A)", loc="left", fontweight="normal")
        if completed or pruned:
            ax[0, 0].legend(frameon=True, framealpha=0.95, fontsize=8)

        # B) Hyperband trajectories and resource rungs.
        best_number = study.best_trial.number if completed else None
        state_styles = {
            TrialState.COMPLETE: (blue, "-", "Completed"),
            TrialState.PRUNED: (orange, "--", "Pruned"),
            TrialState.FAIL: (red, ":", "Failed"),
        }
        plotted_states = set()
        for trial in study.trials:
            points = [
                (step, float(value))
                for step, value in sorted(trial.intermediate_values.items())
                if np.isfinite(value)
            ]
            if not points:
                continue

            epochs, values = zip(*points)
            color, linestyle, _ = state_styles.get(
                trial.state, ("#98A2B3", "--", "Other")
            )
            is_best = trial.number == best_number
            ax[0, 1].plot(
                epochs,
                values,
                color=navy if is_best else color,
                linestyle="-" if is_best else linestyle,
                linewidth=2.8 if is_best else 1.0,
                alpha=1.0 if is_best else 0.28,
                zorder=5 if is_best else 2,
            )
            ax[0, 1].scatter(
                epochs[-1],
                values[-1],
                color=navy if is_best else color,
                s=28 if is_best else 12,
                alpha=1.0 if is_best else 0.55,
                zorder=6,
            )
            plotted_states.add(trial.state)

        for rung in [5, 15, 45]:
            if rung <= MAX_TUNING_EPOCHS:
                ax[0, 1].axvline(rung, color="#CBD5E1", linestyle=":", linewidth=1)
                ax[0, 1].text(
                    rung,
                    0.98,
                    f"r={rung}",
                    transform=ax[0, 1].get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color="#667085",
                )

        handles = []
        if best_number is not None:
            handles.append(Line2D([0], [0], color="navy", linestyle="-", lw=2.8, label="Best trial"))
        handles.extend(
            Line2D(
                [0],
                [0],
                color=state_styles[state][0], 
                linestyle=state_styles[state][1],
                label=state_styles[state][2],
            )
            for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
            if state in plotted_states
        )
        ax[0, 1].set(
            xlabel="Training resource (epoch)",
            ylabel=r"Validation RMSE ($\mathrm{m^3/s}$)",
        )
        ax[0, 1].set_title("B)", loc="left", fontweight="normal")
        if handles:
            ax[0, 1].legend(handles=handles, frameon=True, framealpha=0.95, fontsize=8)

        # C) fANOVA importance as a gradient lollipop chart.
        if importances:
            selected = list(importances.items())[:9][::-1]
            names = [format_parameter(name) for name, _ in selected]
            values = np.asarray([value for _, value in selected])
            positions = np.arange(len(selected))
            colors = plt.cm.turbo_r(Normalize(0, values.max() + 1e-12)(values))

            for position, value, color in zip(positions, values, colors):
                ax[1, 0].hlines(position, 0, value, color=color, linewidth=3.2, alpha=0.75)
                ax[1, 0].scatter(
                    value,
                    position,
                    s=115,
                    color=color,
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=4,
                )
                ax[1, 0].text(
                    value + values.max() * 0.025,
                    position,
                    f"{100 * value:.1f}%",
                    va="center",
                    fontsize=7.5,
                    color="#475467",
                )

            ax[1, 0].set(
                yticks=positions,
                yticklabels=names,
                xlabel="Relative importance",
                xlim=(0, values.max() * 1.18),
            )
            top_parameters = list(importances)[:2]
        else:
            ax[1, 0].text(
                0.5,
                0.5,
                "Parameter importance requires\nat least two completed trials",
                ha="center",
                va="center",
                color="#667085",
                transform=ax[1, 0].transAxes,
            )
            top_parameters = ["learning_rate", "diffusion_weight"]
        ax[1, 0].set_title("C)", loc="left", fontweight="normal")

        # D) Joint landscape of the two most influential parameters.
        landscape_trials = [
            trial
            for trial in completed
            if all(parameter in trial.params for parameter in top_parameters)
        ]
        if landscape_trials and len(top_parameters) == 2:
            x_raw = [trial.params[top_parameters[0]] for trial in landscape_trials]
            y_raw = [trial.params[top_parameters[1]] for trial in landscape_trials]
            x_values, x_categories, x_log = encode_parameter(
                x_raw, top_parameters[0], jitter=True
            )
            y_values, y_categories, y_log = encode_parameter(
                y_raw, top_parameters[1], jitter=True
            )
            rmse = np.asarray([last_finite_value(trial) for trial in landscape_trials])
            epochs = np.asarray(
                [trial.user_attrs.get("epochs_completed", MAX_TUNING_EPOCHS) for trial in landscape_trials]
            )
            sizes = 45 + 125 * epochs / max(epochs.max(), 1)
            norm = Normalize(vmin=rmse.min(), vmax=rmse.max() + 1e-12)
            scatter = ax[1, 1].scatter(
                x_values,
                y_values,
                c=rmse,
                norm=norm,
                cmap="turbo_r",
                s=sizes,
                edgecolor="white",
                linewidth=0.85,
                alpha=0.88,
                zorder=3,
            )
            best_index = int(np.argmin(rmse))
            ax[1, 1].scatter(
                x_values[best_index],
                y_values[best_index],
                marker="*",
                s=280,
                color="#FFD166",
                edgecolor=navy,
                linewidth=1.2,
                zorder=5,
            )

            if x_categories is not None:
                ax[1, 1].set_xticks(range(len(x_categories)), [str(value) for value in x_categories])
            elif x_log:
                ax[1, 1].set_xscale("log")
            if y_categories is not None:
                ax[1, 1].set_yticks(range(len(y_categories)), [str(value) for value in y_categories])
            elif y_log:
                ax[1, 1].set_yscale("log")

            colorbar = fig.colorbar(scatter, ax=ax[1, 1], fraction=0.046, pad=0.025)
            colorbar.set_label(r"Validation RMSE ($\mathrm{m^3/s}$)", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
            colorbar.outline.set_visible(False)
            ax[1, 1].text(
                0.98,
                0.03,
                "Bubble size = epochs",
                transform=ax[1, 1].transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#667085",
            )
            ax[1, 1].set(
                xlabel=format_parameter(top_parameters[0]),
                ylabel=format_parameter(top_parameters[1]),
            )
        else:
            ax[1, 1].text(
                0.5,
                0.5,
                "The performance landscape requires\ncompleted trials with two shared parameters",
                ha="center",
                va="center",
                color="#667085",
                transform=ax[1, 1].transAxes,
            )
        ax[1, 1].set_title("D)", loc="left", fontweight="normal")

        fig.tight_layout()
        fig.savefig(HYPERTUNING_PLOT_PDF, bbox_inches="tight")
        #fig.savefig(HYPERTUNING_PLOT_PNG, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    plot_parallel_coordinates(completed, importances)


def run_hypertuning(series, train_end, validation_end, flow_mean, flow_std):
    # GPT-2 dimensions, attention heads, and layers are fixed by the frozen
    # pretrained backbone and are therefore excluded from this search space.
    sampler = TPESampler(
        seed=TUNING_SEED,
        multivariate=True,
        n_startup_trials=20,
    )
    pruner = HyperbandPruner(
        min_resource=5,
        max_resource=MAX_TUNING_EPOCHS,
        reduction_factor=3,
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    counted_trials = sum(
        trial.state in (TrialState.COMPLETE, TrialState.PRUNED)
        for trial in study.trials
    )
    remaining_trials = max(0, N_TRIALS - counted_trials)

    if remaining_trials:
        study.optimize(
            make_objective(
                series,
                train_end,
                validation_end,
                flow_mean,
                flow_std,
            ),
            n_trials=remaining_trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )

    study.trials_dataframe().to_csv(TRIALS_FILE, index=False)
    plot_hypertuning_analysis(study)
    best = {
        "study_name": STUDY_NAME,
        "sampler": "Multivariate TPE",
        "pruner": "Hyperband",
        "best_trial": study.best_trial.number,
        "best_validation_RMSE_m3_s": study.best_value,
        "best_parameters": study.best_params,
    }

    with open(BEST_PARAMETERS_FILE, "w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)

    print(f"\nBest validation RMSE: {study.best_value:.2f} m^3/s")
    print(json.dumps(study.best_params, indent=2))
    return study.best_params, study.best_value


def summarize_results(results_df, best_validation_rmse):
    metrics = [
        "RMSE",
        "MAE",
        "SMAPE",
        "Training_time_s",
        "Inference_time_s",
        "Inference_time_per_window_s",
    ]
    summary = {
        "Architecture": MODEL_NAME,
        "Parameters": int(results_df["Parameters"].iloc[0]),
        "Best_validation_RMSE": best_validation_rmse,
    }

    for metric in metrics:
        summary[f"{metric}_mean"] = results_df[metric].mean()
        summary[f"{metric}_std"] = results_df[metric].std()

    return pd.DataFrame([summary])


def run_final_evaluation(
    config,
    best_validation_rmse,
    series,
    train_end,
    validation_end,
    flow_mean,
    flow_std,
):
    train, _, test = make_datasets(
        series,
        train_end,
        validation_end,
        config["lookback"],
    )
    test_loader = make_loader(test, config["batch_size"])
    results = []

    for seed in FINAL_SEEDS:
        set_seed(seed)
        train_loader = make_loader(
            train,
            config["batch_size"],
            shuffle=True,
            seed=seed,
        )
        model = FlowDiffusionLLM(config).to(DEVICE)
        optimizer, parameters = make_optimizer(model, config)
        start = time.perf_counter()

        for _ in range(FINAL_EPOCHS):
            train_one_epoch(model, train_loader, optimizer, parameters, config)

        training_time = time.perf_counter() - start
        metrics = evaluate(model, test_loader, flow_mean, flow_std)
        results.append(
            {
                "Architecture": MODEL_NAME,
                "Seed": seed,
                "Parameters": sum(p.numel() for p in parameters),
                "Training_time_s": training_time,
                **metrics,
            }
        )
        print(
            f"seed={seed:2d} "
            f"RMSE={metrics['RMSE']:.2f} "
            f"MAE={metrics['MAE']:.2f} "
            f"SMAPE={metrics['SMAPE']:.2f}%"
        )

        del model
        clear_memory()

    results_df = pd.DataFrame(results)
    summary_df = summarize_results(results_df, best_validation_rmse)
    results_df.to_csv(FINAL_RUNS_FILE, index=False)
    summary_df.to_csv(FINAL_SUMMARY_FILE, index=False)
    print("\n", summary_df.to_string(index=False))


def main():
    series, train_end, validation_end, flow_mean, flow_std = load_flow()
    best_config, best_validation_rmse = run_hypertuning(
        series,
        train_end,
        validation_end,
        flow_mean,
        flow_std,
    )
    run_final_evaluation(
        best_config,
        best_validation_rmse,
        series,
        train_end,
        validation_end,
        flow_mean,
        flow_std,
    )


if __name__ == "__main__":
    main()

