import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Synthetic load components
np.random.seed(1)
n_samples = 1000
t = np.arange(n_samples)
base_load = 500
trend = 0.02 * t
short_cycle = 80 * np.sin(2 * np.pi * t / 60)
long_cycle = 150 * np.sin(2 * np.pi * t / 600)
noise = np.random.normal(0, 25, n_samples)
load = base_load + trend + short_cycle + long_cycle + noise

# Add occasional consumption peaks and reductions
for start in [700, 1800, 3200, 4200]:
    load[start:start + 100] += 150
for start in [1200, 2700, 3800]:
    load[start:start + 80] -= 100
load = np.maximum(load, 0)

# Save Example
df_example = pd.DataFrame({"DateTime": pd.date_range(start="2026-01-01 00:00:00", periods=n_samples, freq="s"), "Conso": load})
df_example.to_csv("miris_load.csv", index=False)
print(f"\nCreated miris_load.csv with {len(df_example)} samples.")

plt.figure(figsize=(8, 3))
plt.plot(np.arange(len(df_example)), df_example["Conso"], "k", linewidth=0.8)
plt.xlim(0, len(df_example)-1)
plt.xlabel("Samples")
plt.ylabel("Load")
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.show()

###############################################################################

import os, time, warnings, contextlib
os.environ["NIXTLA_ID_AS_COL"] = "1"
import logging
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, TFT, RNN, DilatedRNN, NHITS, TCN, BiTCN, LSTM, NBEATS, NBEATSx, GRU, Informer, TiDE, PatchTST, FEDformer, DeepAR, TimesNet

horizon, input_size, epochs, test_ratio, freq = 5, 100, 10, 0.2, "s"

raw = pd.read_csv("miris_load.csv", parse_dates=["DateTime"])
df = pd.DataFrame({"unique_id": "Series1", "ds": raw["DateTime"], "y": raw["Conso"].astype(float)})
split = int(len(df)*(1-test_ratio))
train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

def metrics(y, p):
    rmse = np.sqrt(np.mean((y-p)**2))
    mae = mean_absolute_error(y, p)
    smape = 100*np.mean(2*np.abs(y-p)/np.maximum(np.abs(y)+np.abs(p), 1e-10))
    msle = mean_squared_log_error(np.clip(y, 0, None), np.clip(p, 0, None))
    return rmse, mae, smape, msle

def evaluate(model_class):
    name = model_class.__name__
    model = model_class(input_size=input_size, h=horizon, max_steps=epochs, random_seed=0,
                        enable_progress_bar=False, logger=False)
    nf = NeuralForecast(models=[model], freq=freq)

    with open(os.devnull, "w") as null, contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
        start = time.perf_counter()
        nf.fit(df=train, val_size=0)
        train_time = time.perf_counter()-start

        history, window_metrics = train.copy(), []
        start = time.perf_counter()
        for i in range(0, len(test), horizon):
            block = test.iloc[i:i+horizon]
            pred = nf.predict(df=history).reset_index()[name].to_numpy(float)[:len(block)]
            window_metrics.append(metrics(block["y"].to_numpy(float), pred))
            history = pd.concat([history, block], ignore_index=True)
        test_time = time.perf_counter()-start

    values = np.array(window_metrics)
    ddof = 1 if len(values) > 1 else 0
    return values.mean(0), values.std(0, ddof=ddof), train_time, test_time

models = [MLP, TFT, RNN, DilatedRNN, NHITS, TCN, BiTCN, LSTM, NBEATS, 
          GRU, Informer, TiDE, PatchTST, FEDformer, DeepAR, TimesNet]

results = []
for model in models:
    try:
        mean, std, train_time, test_time = evaluate(model)
        results.append({
            "Model": model.__name__,
            "RMSE_mean": mean[0], "RMSE_std": std[0],
            "MAE_mean": mean[1], "MAE_std": std[1],
            "SMAPE_mean": mean[2], "SMAPE_std": std[2],
            "MSLE_mean": mean[3], "MSLE_std": std[3],
            "Train_time": train_time, "Test_time": test_time,
            "Error": ""
        })
    except Exception as error:
        results.append({"Model": model.__name__, "Error": str(error)})

output_file = "forecast_results.csv"
pd.DataFrame(results).to_csv(output_file, index=False)

###############################################################################
results_df = pd.read_csv(output_file)
valid = results_df.dropna(subset=["RMSE_mean"])

print(r"Model & RMSE & MAE & SMAPE (\%) & Train (s) & Test (s) \\")
for _, row in valid.iterrows():
    values = " & ".join(
        f"{row[f'{metric}_mean']:.2E} $\\pm$ {row[f'{metric}_std']:.2E}"
        for metric in ["RMSE", "MAE", "SMAPE"]
    )
    print(f"{row['Model']} & {values} & {row['Train_time']:.2E} & {row['Test_time']:.2E} \\\\")

failed = results_df[results_df["Error"].fillna("") != ""]
if not failed.empty:
    print("\nFailed models:")
    for _, row in failed.iterrows():
        print(f"{row['Model']}: {row['Error']}")
