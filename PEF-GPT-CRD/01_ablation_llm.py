gpu_use = True
gpu_number = "4"
if gpu_use == True:
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda:0")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(f"GPU {gpu_number}")

import os, gc, random, time, warnings
os.environ.update({"USE_TF": "0", "USE_FLAX": "0", "USE_TORCH": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DATA_FILE, FLOW_COLUMN, RAIN_COLUMN = "tucurui.csv", "Natural Flow", "UPH610010000"
TRAIN_RATIO, LOOKBACK, HORIZON, PATCH_SIZE = 0.8, 512, 7, 8
LLM_NAME, DIFFUSION_STEPS, DIFFUSION_WEIGHT, DIFFUSION_SAMPLES = "gpt2", 100, 0.1, 10
BATCH_SIZE, LEARNING_RATE, EPOCHS, SEEDS = 32, 1e-3, 100, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

def make_windows(features, target):
    n = len(target) - LOOKBACK - HORIZON + 1
    if n <= 0: raise ValueError("The series is shorter than LOOKBACK + HORIZON.")
    x = np.stack([features[i:i + LOOKBACK] for i in range(n)])
    y = np.stack([target[i + LOOKBACK:i + LOOKBACK + HORIZON] for i in range(n)])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__(); self.dimension = dimension; self.mlp = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
    def forward(self, t):
        half = self.dimension // 2; scale = np.log(10000) / max(half - 1, 1)
        frequency = torch.exp(-scale * torch.arange(half, device=t.device)); angle = t.float()[:, None] * frequency[None]
        embedding = torch.cat((angle.sin(), angle.cos()), dim=1)
        if embedding.shape[1] < self.dimension: embedding = nn.functional.pad(embedding, (0, 1))
        return self.mlp(embedding)

class PatchEncoder(nn.Module):
    def __init__(self, input_channels, dimension, use_patches=True):
        super().__init__(); self.use_patches = use_patches; width = PATCH_SIZE * input_channels if use_patches else input_channels
        if use_patches and LOOKBACK % PATCH_SIZE: raise ValueError("LOOKBACK must be divisible by PATCH_SIZE.")
        self.projection = nn.Sequential(nn.Linear(width, dimension), nn.LayerNorm(dimension))
    def forward(self, x):
        if self.use_patches:
            b, length, channels = x.shape; x = x.reshape(b, length // PATCH_SIZE, PATCH_SIZE * channels)
        return self.projection(x)

class ForecastBackbone(nn.Module):
    def __init__(self, use_llm=True, use_positions=True):
        super().__init__(); self.use_llm, self.use_positions = use_llm, use_positions
        if use_llm:
            self.network = GPT2Model.from_pretrained(LLM_NAME); self.dimension = self.network.config.n_embd
            for parameter in self.network.parameters(): parameter.requires_grad = False
        else:
            self.dimension = GPT2Model.from_pretrained(LLM_NAME).config.n_embd
            self.network = nn.Sequential(nn.Linear(self.dimension, self.dimension), nn.GELU(), nn.Linear(self.dimension, self.dimension))
        self.head = nn.Sequential(nn.LayerNorm(self.dimension), nn.Linear(self.dimension, HORIZON))
    def forward(self, tokens):
        if self.use_llm:
            self.network.eval(); b, length, _ = tokens.shape
            positions = torch.arange(length, device=tokens.device)[None].expand(b, -1) if self.use_positions else torch.zeros(b, length, dtype=torch.long, device=tokens.device)
            hidden = self.network(inputs_embeds=tokens, position_ids=positions, return_dict=True).last_hidden_state
        else: hidden = self.network(tokens)
        condition = hidden.mean(1)
        return self.head(condition), condition

class ResidualDenoiser(nn.Module):
    def __init__(self, condition_dimension, use_condition=True, use_time=True, use_skip=True):
        super().__init__(); self.use_condition, self.use_time, self.use_skip = use_condition, use_time, use_skip
        self.time = SinusoidalTimeEmbedding(condition_dimension) if use_time else None
        input_dimension = HORIZON + condition_dimension * (int(use_condition) + int(use_time))
        self.input = nn.Linear(input_dimension, 256)
        self.blocks = nn.Sequential(nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.output = nn.Linear(256, HORIZON)
    def forward(self, noisy_residual, condition, t):
        parts = [noisy_residual]
        if self.use_condition: parts.append(condition)
        if self.use_time: parts.append(self.time(t))
        hidden = self.input(torch.cat(parts, 1)); transformed = self.blocks(hidden)
        return self.output(transformed + hidden if self.use_skip else transformed)

class DiffusionLLM(nn.Module):
    def __init__(self, input_channels, use_diffusion=True, use_condition=True, use_time=True, use_skip=True, use_positions=True, use_patches=True, use_llm=True):
        super().__init__(); self.use_diffusion = use_diffusion
        self.backbone = ForecastBackbone(use_llm, use_positions); dimension = self.backbone.dimension
        self.encoder = PatchEncoder(input_channels, dimension, use_patches)
        if use_diffusion:
            self.denoiser = ResidualDenoiser(dimension, use_condition, use_time, use_skip)
            beta = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS); alpha = 1 - beta; alpha_bar = torch.cumprod(alpha, 0)
            self.register_buffer("beta", beta); self.register_buffer("alpha", alpha); self.register_buffer("alpha_bar", alpha_bar)
    def encode(self, x): return self.backbone(self.encoder(x))
    def diffusion_loss(self, residual, condition):
        t = torch.randint(DIFFUSION_STEPS, (residual.shape[0],), device=residual.device); noise = torch.randn_like(residual); abar = self.alpha_bar[t, None]
        noisy = abar.sqrt() * residual + (1 - abar).sqrt() * noise
        return nn.functional.mse_loss(self.denoiser(noisy, condition, t), noise)
    def forward(self, x, y=None):
        base, condition = self.encode(x)
        if y is None: return base
        forecast_loss = nn.functional.mse_loss(base, y)
        residual = (y - base).detach()
        diffusion_loss = self.diffusion_loss(residual, condition) if self.use_diffusion else base.new_zeros(())
        return forecast_loss, diffusion_loss
    @torch.inference_mode()
    def predict(self, x, samples=DIFFUSION_SAMPLES):
        base, condition = self.encode(x)
        if not self.use_diffusion: return base
        estimates = []
        for _ in range(samples):
            residual = torch.randn_like(base)
            for step in reversed(range(DIFFUSION_STEPS)):
                t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long); predicted_noise = self.denoiser(residual, condition, t)
                residual = (residual - self.beta[step] / torch.sqrt(1 - self.alpha_bar[step]) * predicted_noise) / torch.sqrt(self.alpha[step])
                if step: residual += self.beta[step].sqrt() * torch.randn_like(residual)
            estimates.append(base + residual)
        return torch.stack(estimates).mean(0)

def train_model(model, loader):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE, weight_decay=1e-4); history = []
    for _ in range(EPOCHS):
        model.train(); running = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE); optimizer.zero_grad(set_to_none=True); forecast_loss, diffusion_loss = model(x, y)
            loss = forecast_loss + DIFFUSION_WEIGHT * diffusion_loss; loss.backward(); nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0); optimizer.step(); running += forecast_loss.item()
        history.append(running / len(loader))
    return history

def evaluate(model, loader, target_mean, target_std):
    model.eval(); predictions, targets = [], []; start = time.perf_counter()
    for x, y in loader: predictions.append(model.predict(x.to(DEVICE)).cpu()); targets.append(y)
    inference_time = time.perf_counter() - start
    predictions = torch.cat(predictions).numpy().ravel() * target_std + target_mean; targets = torch.cat(targets).numpy().ravel() * target_std + target_mean; error = predictions - targets
    return {"RMSE": np.sqrt(np.mean(error ** 2)), "MAE": np.mean(np.abs(error)), "SMAPE": 100 * np.mean(2 * np.abs(error) / (np.abs(targets) + np.abs(predictions) + 1e-8)), "Inference_time_s": inference_time}

data = pd.read_csv(DATA_FILE, sep=";", decimal=",")[[FLOW_COLUMN, RAIN_COLUMN]].dropna().astype(np.float32)
split = int(len(data) * TRAIN_RATIO); training = data.iloc[:split]; means, stds = training.mean(), training.std(ddof=0).replace(0, 1)
normalized = (data - means) / stds; features = normalized[[FLOW_COLUMN, RAIN_COLUMN]].to_numpy(); target = normalized[FLOW_COLUMN].to_numpy()
x_train, y_train = make_windows(features[:split], target[:split]); x_test, y_test = make_windows(features[split - LOOKBACK:], target[split - LOOKBACK:])
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

BASE = dict(use_diffusion=True, use_condition=True, use_time=True, use_skip=True, use_positions=True, use_patches=True, use_llm=True)
CONFIGURATIONS = {
    "Full model": BASE,
    "Without diffusion": {**BASE, "use_diffusion": False},
    "Unconditional diffusion": {**BASE, "use_condition": False},
    "Without time embedding": {**BASE, "use_time": False},
    "Without denoiser skip": {**BASE, "use_skip": False},
    "Without position indices": {**BASE, "use_positions": False},
    "Without patching": {**BASE, "use_patches": False},
    "Without precipitation": {**BASE, "input_channels": 1},
    "MLP backbone": {**BASE, "use_llm": False}
}

results, histories = [], {}
for architecture, config in CONFIGURATIONS.items():
    histories[architecture] = []
    for seed in SEEDS:
        set_seed(seed); channels = config.get("input_channels", 2); model_config = {k: v for k, v in config.items() if k != "input_channels"}
        current_x_train = x_train[..., :channels]; current_x_test = x_test[..., :channels]
        generator = torch.Generator().manual_seed(seed); train_loader = DataLoader(TensorDataset(current_x_train, y_train), batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        model = DiffusionLLM(channels, **model_config).to(DEVICE); start = time.perf_counter(); history = train_model(model, train_loader); training_time = time.perf_counter() - start
        metrics = evaluate(model, DataLoader(TensorDataset(current_x_test, y_test), batch_size=BATCH_SIZE), means[FLOW_COLUMN], stds[FLOW_COLUMN])
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad); results.append({"Architecture": architecture, "Seed": seed, "Parameters": parameters, "Training_time_s": training_time, **metrics}); histories[architecture].append(history)
        print(f"{architecture:28s} seed={seed} RMSE={metrics['RMSE']:.2f} MAE={metrics['MAE']:.2f} SMAPE={metrics['SMAPE']:.2f}%")
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

results_df = pd.DataFrame(results); metrics = ["RMSE", "MAE", "SMAPE", "Training_time_s", "Inference_time_s"]
summary_df = results_df.groupby("Architecture", as_index=False).agg(Parameters=("Parameters", "first"), **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")}).sort_values("RMSE_mean").reset_index(drop=True)
full_rmse = summary_df.loc[summary_df.Architecture.eq("Full model"), "RMSE_mean"].iloc[0]; summary_df["RMSE_change_percent"] = 100 * (summary_df.RMSE_mean - full_rmse) / full_rmse
results_df.to_csv("ablation_individual_runs.csv", index=False); summary_df.to_csv("ablation_summary.csv", index=False); print("\n", summary_df.to_string(index=False))

order = summary_df.Architecture.tolist(); colors = dict(zip(order, plt.cm.turbo(np.linspace(0, 1, len(order))))); fig, ax = plt.subplots(2, 2, figsize=(13, 9))
plot_df = summary_df.sort_values("RMSE_mean", ascending=False); pos = np.arange(len(plot_df)); ax[0, 0].barh(pos, plot_df.RMSE_mean, xerr=plot_df.RMSE_std.fillna(0), color=[colors[x] for x in plot_df.Architecture], capsize=4); ax[0, 0].set(yticks=pos, yticklabels=plot_df.Architecture, xlabel="RMSE"); ax[0, 0].set_title("A)", loc="left")
plot_df = summary_df[summary_df.Architecture.ne("Full model")].sort_values("RMSE_change_percent"); pos = np.arange(len(plot_df)); ax[0, 1].barh(pos, plot_df.RMSE_change_percent, color=[colors[x] for x in plot_df.Architecture]); ax[0, 1].axvline(0, color="black", ls="--"); ax[0, 1].set(yticks=pos, yticklabels=plot_df.Architecture, xlabel="RMSE change relative to full model (%)"); ax[0, 1].set_title("B)", loc="left")
for _, row in summary_df.iterrows(): ax[1, 0].scatter(row.Parameters, row.RMSE_mean, s=90, color=colors[row.Architecture], edgecolor="black", label=row.Architecture)
ax[1, 0].set(xlabel="Trainable parameters", ylabel="RMSE"); ax[1, 0].set_title("C)", loc="left"); ax[1, 0].legend(fontsize=7)
for name in order:
    curves = np.asarray(histories[name]); epochs = np.arange(1, EPOCHS + 1); ax[1, 1].plot(epochs, curves.mean(0), color=colors[name], label=name); ax[1, 1].fill_between(epochs, curves.mean(0) - curves.std(0), curves.mean(0) + curves.std(0), color=colors[name], alpha=0.12)
ax[1, 1].set(xlabel="Epoch", ylabel="Forecasting loss"); ax[1, 1].set_title("D)", loc="left"); ax[1, 1].legend(fontsize=7)
for axis in ax.ravel(): axis.grid(True, alpha=0.25)
plt.tight_layout(); plt.savefig("ablation_study.pdf", bbox_inches="tight"); plt.savefig("ablation_study.png", dpi=300, bbox_inches="tight"); plt.show()

best_rmse = summary_df.RMSE_mean.min(); selected = summary_df[summary_df.RMSE_mean <= 1.01 * best_rmse].sort_values(["Parameters", "RMSE_mean"]).iloc[0]
print("\nSelected architecture\n", selected.to_string())
