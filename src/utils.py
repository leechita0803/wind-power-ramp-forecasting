import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import METRICS_DIR, PLOTS_DIR


def evaluate_forecast(y_true, y_pred):
    """
    Calculate forecast evaluation metrics.
    """
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }
    return metrics


def save_metrics(metrics, filename="metrics.json"):
    """
    Save evaluation metrics as a JSON file.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    save_path = METRICS_DIR / filename
    print("Saving metrics to:", save_path)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics file saved successfully.")


def save_forecast_plot(y_true, y_pred, filename="actual_vs_predicted.png", n_points=200):
    """
    Save plot of actual vs predicted values.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / filename

    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index[:n_points], y_true.values[:n_points], label="Actual")
    plt.plot(y_true.index[:n_points], y_pred[:n_points], label="Forecast")
    plt.xlabel("Time")
    plt.ylabel("LV ActivePower (kW)")
    plt.title("Actual vs Forecasted Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("Forecast plot saved to:", save_path)


def save_imf_prediction_plots(test_index, imf_predictions, filename_prefix="imf_prediction"):
    """
    Save one plot for each predicted IMF.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for imf_name, values in imf_predictions.items():
        save_path = PLOTS_DIR / f"{filename_prefix}_{imf_name}.png"

        plt.figure(figsize=(12, 4))
        plt.plot(test_index[:300], values[:300], label=f"Predicted {imf_name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Prediction for {imf_name.upper()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"{imf_name} plot saved to: {save_path}")


def save_mode_plot(series, filename="mode_2_plot.png", title="Mode 2", ylabel="Mode 2 Value"):
    """
    Save a single mode/component plot.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / filename

    plt.figure(figsize=(12, 3))
    if hasattr(series, "index"):
        plt.plot(series.index, series.values)
    else:
        plt.plot(series)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("Mode plot saved to:", save_path)