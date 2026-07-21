print("main.py started")

from pathlib import Path
import json
import pandas as pd
import os

from src.preprocess import load_and_preprocess_data, split_data
from src.decompose import decompose_signal
from src.train import train_models, predict_models
from src.integrate import integrate_predictions
from src.ramp_detection import (
    detect_ramps,
    calculate_ramp_scores,
    save_ramp_plot
)
from src.utils import (
    evaluate_forecast,
    save_metrics,
    save_forecast_plot,
    save_imf_prediction_plots,
    save_mode_plot
)
from src.config import TARGET_COL, METRICS_DIR


def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    print("Data loaded successfully.")
    print(df.head())
    print(df.shape)

    train_df, test_df = split_data(df)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("Running decomposition on training target...")
    train_imfs = decompose_signal(train_df[TARGET_COL])

    print("Decomposition successful.")
    print(train_imfs.head())
    print(train_imfs.shape)

    print("Preparing training and testing features...")
    feature_cols = [col for col in train_df.columns if col != TARGET_COL]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("Training IMF models...")
    models = train_models(X_train, train_imfs)
    print("Models trained successfully.")

    print("Predicting IMF components on test data...")
    imf_predictions = predict_models(models, X_test)
    print("Prediction successful.")
    print("First 5 IMF1 predictions:", imf_predictions["imf1"][:5])

    print("Saving IMF prediction plots...")
    save_imf_prediction_plots(test_df.index, imf_predictions)

    print("Integrating IMF predictions into final forecast...")
    final_pred = integrate_predictions(imf_predictions)
    print("Integration successful.")
    print("First 5 final predictions:", final_pred[:5])
    print("Total number of final predictions:", len(final_pred))

    print("Evaluating final forecast...")
    y_test = test_df[TARGET_COL]
    metrics = evaluate_forecast(y_test, final_pred)

    print("Evaluation successful.")
    print("Metrics:", metrics)

    print("Saving metrics...")
    save_metrics(metrics)

    print("Saving forecast plot...")
    save_forecast_plot(y_test, final_pred, filename="actual_vs_predicted.png")

    # -----------------------------
    # Save forecast comparison CSV
    # -----------------------------

    PREDICTIONS_DIR = Path("outputs/predictions")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame({
        "Timestamp": y_test.index,
        "Actual": y_test.values,
        "Forecast": final_pred
    })

    comparison_df["Absolute Error"] = (
        comparison_df["Actual"] - comparison_df["Forecast"]
    ).abs()

    comparison_df.to_csv(
    PREDICTIONS_DIR / "forecast.csv",
    index=False
    )

    print("Forecast comparison saved successfully.")

    if "imf2" in train_imfs.columns:
        print("Saving mode 2 plot...")
        save_mode_plot(
            train_imfs["imf2"],
            filename="mode_2_plot.png",
            title="Mode 2",
            ylabel="Mode 2 Value"
        )

    print("Preparing actual and predicted dataframes for ramp detection...")
    actual_data = pd.DataFrame({"Og": y_test.values}, index=y_test.index)
    predicted_data = pd.DataFrame({"Pred": final_pred}, index=y_test.index)

    installed_capacity = 1
    ramp_threshold_percentage = 15
    ramp_duration_minutes = 120
    interval_minutes = 10
    ramp_duration_intervals = ramp_duration_minutes // interval_minutes

    print("Detecting ramps in actual data...")
    ramps_actual = detect_ramps(
        actual_data["Og"],
        threshold_percentage=ramp_threshold_percentage,
        duration_intervals=ramp_duration_intervals,
        installed_capacity=installed_capacity
    )

    print("Detecting ramps in predicted data...")
    ramps_predicted = detect_ramps(
        predicted_data["Pred"],
        threshold_percentage=ramp_threshold_percentage,
        duration_intervals=ramp_duration_intervals,
        installed_capacity=installed_capacity
    )

    print("Saving ramp event tables...")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ramps_actual.to_csv(PREDICTIONS_DIR / "ramps_actual.csv",index=False)

    ramps_predicted.to_csv(PREDICTIONS_DIR / "ramps_predicted.csv",index=False)

    print("Calculating ramp detection scores...")
    ramp_scores = calculate_ramp_scores(
        ramps_actual,
        ramps_predicted,
        tolerance_minutes=10
    )

    print("Ramp Scores:", ramp_scores)

    with open(METRICS_DIR / "ramp_scores.json", "w") as f:
        json.dump(ramp_scores, f, indent=4)

    print("Saving ramp events plot...")
    save_ramp_plot(
        actual_data=actual_data,
        predicted_data=predicted_data,
        ramps_actual=ramps_actual,
        ramps_predicted=ramps_predicted,
        filename="ramp_events_plot.png"
    )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()