gpu_use = True
complete = True
generalization_analisys = True

gpu_number = "0"
if gpu_use == True:
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda:0")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(f"GPU {gpu_number}")

if complete: # Complete experiment
    EPOCHS = 100
    RUNS = 1
else:
    EPOCHS = 2
    RUNS = 1
    
# AMUHBB - Norte
# AMCLR - Sudeste/Centro-Oeste 
# RIGAR - Sul
# JEUITP - Nordeste

from pathlib import Path
if generalization_analisys == True:
    OUTPUT_DIR = Path("Results_04_benchmarking_generalization")
    DATA_DIR = Path("dados_hidrologicos")
    RESERVOIR_ID = "IGUHBI"
    FLOW_COLUMN = "val_vazaoafluente"
else:
    OUTPUT_DIR = Path("Results_04_benchmarking")
    DATA_PATH = "tucurui.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import os
os.environ["NIXTLA_ID_AS_COL"] = "1"

import time
import random
import warnings
import logging
import contextlib
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, TFT, RNN, DilatedRNN, NHITS, TCN, BiTCN, LSTM, NBEATS, NBEATSx, GRU, Informer, TiDE, PatchTST, FEDformer, DeepAR, TimesNet, DLinear, NLinear, VanillaTransformer, Autoformer, StemGNN, HINT, TimeLLM, TSMixer, TSMixerx, MLPMultivariate, iTransformer, DeepNPTS, SOFTS, TimeMixer, KAN, RMoK, TimeXer, xLSTM

logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

DATA_PATH = "tucurui.csv"
TARGET_COLUMN = "Natural Flow"
LOOKBACK = 64
HORIZON = 7
MAX_STEPS = EPOCHS
BATCH_SIZE = 16
TEST_RATIO = 0.20
SEEDS = list(range(1, RUNS+1))
FREQ = 1

MODELS = [MLP, TFT, RNN, DilatedRNN, NHITS, TCN, BiTCN, LSTM, NBEATS, GRU, Informer, TiDE, PatchTST, FEDformer, DeepAR, TimesNet, DLinear, NLinear, VanillaTransformer, 
	Autoformer,  DeepNPTS, KAN]

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# Metrics
# ============================================================

def calculate_metrics(targets, predictions):
    error = predictions - targets
    return {
        "MSE": np.mean(error**2),
        "RMSE": np.sqrt(np.mean(error**2)),
        "MAE": np.mean(np.abs(error)),
        "SMAPE": 100 * np.mean(
            2 * np.abs(error) /
            (np.abs(targets) + np.abs(predictions) + 1e-8)
        )
    }

# ============================================================
# Load and prepare the dataset
# ============================================================
if generalization_analisys == True:
    files = sorted(DATA_DIR.glob("DADOS_HIDROLOGICOS_HO_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR.resolve()}")
    
    columns = ["id_reservatorio", "nom_reservatorio", "din_instante", FLOW_COLUMN]
    data = pd.concat(
        [pd.read_csv(f, sep=";", decimal=",", usecols=columns) for f in files],
        ignore_index=True,)
    
    data["din_instante"] = pd.to_datetime(data["din_instante"], errors="coerce")
    data[FLOW_COLUMN] = pd.to_numeric(data[FLOW_COLUMN], errors="coerce")
    data = (data[data["id_reservatorio"].eq(RESERVOIR_ID)]
            .dropna(subset=["din_instante", FLOW_COLUMN])
            .sort_values("din_instante")
            .drop_duplicates("din_instante", keep="last")
            .set_index("din_instante"))
    
    # For daily natural/affluent flow, using the daily mean
    data = data.resample("D").agg({FLOW_COLUMN: "mean","nom_reservatorio": 
                                   "first",}).dropna(subset=[FLOW_COLUMN])
    
    print(f"{data['nom_reservatorio'].iloc[0]}: {data.index.min()} to "
          f"{data.index.max()} ({len(data)} observations)")
    
    values = data[FLOW_COLUMN].to_numpy(np.float32)
    
else:
    values = (pd.read_csv(DATA_PATH, sep=";", decimal=",")
              [TARGET_COLUMN].dropna().to_numpy(dtype=np.float32))
    
split = int(len(values) * (1 - TEST_RATIO))
data_mean = values[:split].mean()
data_std = values[:split].std()

if data_std == 0:
    raise ValueError("The training-set standard deviation is zero.")

normalized_values = (values - data_mean) / data_std

# Integer time index, following the sequential sampling of the dataset
data = pd.DataFrame({"unique_id": "Tucurui", 
                     "ds": np.arange(len(normalized_values)),
                     "y": normalized_values})

train = data.iloc[:split].copy()

if len(train) < LOOKBACK + HORIZON:
    raise ValueError("The training set is too short for the selected lookback and horizon.")

if len(data) - split < HORIZON:
    raise ValueError("The test set is shorter than the forecasting horizon.")

print(f"Samples: {len(data)}")
print(f"Training samples: {split}")
print(f"Test samples: {len(data) - split}")
print(f"Test windows: {len(data) - split - HORIZON + 1}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# ============================================================
# Extract the point forecast
# ============================================================

def extract_forecast(forecast, model_name):
    forecast = forecast.reset_index()

    if model_name in forecast.columns:
        column = model_name
    elif f"{model_name}-median" in forecast.columns:
        column = f"{model_name}-median"
    else:
        excluded = {"unique_id", "ds", "cutoff"}
        candidates = [
            column for column in forecast.columns
            if column not in excluded and
            pd.api.types.is_numeric_dtype(forecast[column])
        ]

        median_candidates = [
            column for column in candidates
            if "median" in column.lower()
        ]

        if median_candidates:
            column = median_candidates[0]
        elif candidates:
            column = candidates[0]
        else:
            raise ValueError(
                f"No point-forecast column was found for {model_name}. "
                f"Available columns: {forecast.columns.tolist()}"
            )

    return forecast[column].to_numpy(dtype=float)[:HORIZON]

# ============================================================
# Train and evaluate one model for one seed
# ============================================================

def evaluate_model(model_class, seed):
    set_seed(seed)
    model_name = model_class.__name__

    model = model_class(
        h=HORIZON,
        input_size=LOOKBACK,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        random_seed=seed,
        enable_progress_bar=False,
        logger=False
    )

    nf = NeuralForecast(models=[model], freq=FREQ)

    with open(os.devnull, "w") as null:
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            start = time.perf_counter()
            nf.fit(df=train, val_size=0)
            train_time = time.perf_counter() - start

            parameters = sum(
                parameter.numel()
                for parameter in nf.models[0].parameters()
                if parameter.requires_grad
            )

            predictions, targets = [], []
            start = time.perf_counter()

            # Every possible overlapping seven-step test window
            for origin in range(split, len(data) - HORIZON + 1):
                # Actual observations are available up to the forecast origin
                history = data.iloc[:origin].copy()
                forecast = nf.predict(df=history)
                prediction = extract_forecast(forecast, model_name)

                # Return forecasts to the original m³/s scale
                prediction = prediction * data_std + data_mean
                target = values[origin:origin + HORIZON].astype(float)

                predictions.append(prediction)
                targets.append(target)

            test_time = time.perf_counter() - start

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    metrics = calculate_metrics(targets, predictions)

    del nf, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "Model": model_name,
        "Seed": seed,
        "Parameters": parameters,
        **metrics,
        "Train_time": train_time,
        "Test_time": test_time
    }

# ============================================================
# Evaluate all models and seeds
# ============================================================

results = []
failures = []

for model_class in MODELS:
    model_name = model_class.__name__

    for run, seed in enumerate(SEEDS, 1):
        try:
            result = evaluate_model(model_class, seed)
            results.append(result)

            print(
                f"{model_name:12s} | Run {run:02d}/{len(SEEDS)} | "
                f"Seed {seed:02d} | "
                f"RMSE {result['RMSE']:.6f} | "
                f"MAE {result['MAE']:.6f} | "
                f"SMAPE {result['SMAPE']:.6f}"
            )

        except Exception as error:
            failures.append({
                "Model": model_name,
                "Seed": seed,
                "Error": str(error)
            })
            print(f"{model_name:12s} | Seed {seed:02d} | Failed: {error}")

results_df = pd.DataFrame(results)
failures_df = pd.DataFrame(failures)

results_df.to_csv(OUTPUT_DIR / "benchmark_individual_runs.csv", index=False)
failures_df.to_csv(OUTPUT_DIR / "benchmark_failed_runs.csv", index=False)

# ============================================================
# Mean and standard deviation across seeds
# ============================================================

if results_df.empty:
    raise RuntimeError("No model was evaluated successfully.")

summary_rows = []

for model_name, group in results_df.groupby("Model", sort=False):
    summary_rows.append({
        "Model": model_name,
        "Runs": len(group),
        "Parameters": int(group["Parameters"].iloc[0]),
        "MSE_mean": group["MSE"].mean(),
        "MSE_std": group["MSE"].std(ddof=1),
        "RMSE_mean": group["RMSE"].mean(),
        "RMSE_std": group["RMSE"].std(ddof=1),
        "MAE_mean": group["MAE"].mean(),
        "MAE_std": group["MAE"].std(ddof=1),
        "SMAPE_mean": group["SMAPE"].mean(),
        "SMAPE_std": group["SMAPE"].std(ddof=1),
        "Train_time_mean": group["Train_time"].mean(),
        "Train_time_std": group["Train_time"].std(ddof=1),
        "Test_time_mean": group["Test_time"].mean(),
        "Test_time_std": group["Test_time"].std(ddof=1)
    })

summary_df = (pd.DataFrame(summary_rows).sort_values("RMSE_mean")
    .reset_index(drop=True))
summary_df.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)

# ============================================================
# Display results
# ============================================================

print("\nBenchmark summary:")
print(summary_df.to_string(index=False))

print("\nLaTeX table rows:")
print(r"Model & RMSE & MAE & SMAPE (\%) & Train (s) & Test (s) \\")

for _, row in summary_df.iterrows():
    print(
        f"{row['Model']} & "
        f"{row['RMSE_mean']:.2E} $\\pm$ {row['RMSE_std']:.2E} & "
        f"{row['MAE_mean']:.2E} $\\pm$ {row['MAE_std']:.2E} & "
        f"{row['SMAPE_mean']:.2E} $\\pm$ {row['SMAPE_std']:.2E} & "
        f"{row['Train_time_mean']:.2E} & "
        f"{row['Test_time_mean']:.2E} \\\\"
    )

if not failures_df.empty:
    print("\nFailed runs:")
    print(failures_df.to_string(index=False))
