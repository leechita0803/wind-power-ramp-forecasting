import pandas as pd
import matplotlib.pyplot as plt

from src.config import PLOTS_DIR


def detect_ramps(power_series, threshold_percentage=15, duration_intervals=12, installed_capacity=1):
    """
    Detect ramp events using a rolling window.

    Parameters:
    - power_series: pandas Series with datetime index
    - threshold_percentage: ramp threshold as percentage of installed capacity
    - duration_intervals: number of intervals in the ramp window
    - installed_capacity: normalized installed capacity

    Returns:
    - DataFrame of detected ramp events
    """
    threshold_value = (threshold_percentage / 100) * installed_capacity
    ramps = []

    for i in range(len(power_series) - duration_intervals):
        start_index = i
        end_index = i + duration_intervals

        power_range = power_series.iloc[start_index:end_index + 1]
        min_power = power_range.min()
        max_power = power_range.max()
        power_diff = max_power - min_power

        if power_diff >= threshold_value:
            ramp_direction = "Up" if max_power == power_range.iloc[-1] else "Down"

            ramps.append({
                "Start_Time": power_series.index[start_index],
                "End_Time": power_series.index[end_index],
                "Max_Index": power_range.idxmax(),
                "Min_Index": power_range.idxmin(),
                "Magnitude": float(power_diff),
                "Direction": ramp_direction
            })

    return pd.DataFrame(ramps)


def match_ramps(ramps_actual, ramps_predicted, tolerance_minutes=10):
    """
    Match predicted ramps to actual ramps within a time tolerance.
    """
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    matches = 0
    matched_predicted_indices = set()

    if ramps_actual.empty or ramps_predicted.empty:
        return matches

    for _, actual_ramp in ramps_actual.iterrows():
        actual_start = actual_ramp["Start_Time"]

        for pred_index, predicted_ramp in ramps_predicted.iterrows():
            if pred_index in matched_predicted_indices:
                continue

            predicted_start = predicted_ramp["Start_Time"]

            if abs(actual_start - predicted_start) <= tolerance:
                matches += 1
                matched_predicted_indices.add(pred_index)
                break

    return matches


def calculate_ramp_scores(ramps_actual, ramps_predicted, tolerance_minutes=10):
    """
    Calculate precision, recall, and F1-score for ramp detection.
    """
    matches = match_ramps(ramps_actual, ramps_predicted, tolerance_minutes)

    total_actual = len(ramps_actual)
    total_predicted = len(ramps_predicted)

    precision = matches / total_predicted if total_predicted > 0 else 0
    recall = matches / total_actual if total_actual > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    scores = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "total_actual_ramps": int(total_actual),
        "total_predicted_ramps": int(total_predicted),
        "matches": int(matches)
    }

    return scores


def save_ramp_plot(
    actual_data,
    predicted_data,
    ramps_actual,
    ramps_predicted,
    filename="ramp_events_plot.png",
    n_points=300
):
    """
    Save ramp plot for only the first n_points for speed and readability.

    actual_data: DataFrame with column 'Og'
    predicted_data: DataFrame with column 'Pred'
    ramps_actual: DataFrame of actual ramp events
    ramps_predicted: DataFrame of predicted ramp events
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PLOTS_DIR / filename

    actual_data_plot = actual_data.iloc[:n_points].copy()
    predicted_data_plot = predicted_data.iloc[:n_points].copy()

    start_time = actual_data_plot.index.min()
    end_time = actual_data_plot.index.max()

    ramps_actual_plot = ramps_actual[
        (ramps_actual["Start_Time"] <= end_time) & (ramps_actual["End_Time"] >= start_time)
    ].copy()

    ramps_predicted_plot = ramps_predicted[
        (ramps_predicted["Start_Time"] <= end_time) & (ramps_predicted["End_Time"] >= start_time)
    ].copy()

    plt.figure(figsize=(14, 7))
    plt.plot(actual_data_plot.index, actual_data_plot["Og"], label="Actual Power", color="blue")
    plt.plot(predicted_data_plot.index, predicted_data_plot["Pred"], label="Predicted Power", color="red")

    if not ramps_actual_plot.empty:
        first_actual_idx = ramps_actual_plot.index[0]

        for i, event in ramps_actual_plot.iterrows():
            if event["Max_Index"] in actual_data_plot.index and event["Min_Index"] in actual_data_plot.index:
                plt.scatter(
                    event["Max_Index"],
                    actual_data_plot.loc[event["Max_Index"], "Og"],
                    color="darkred",
                    label="Actual Max Value" if i == first_actual_idx else ""
                )
                plt.scatter(
                    event["Min_Index"],
                    actual_data_plot.loc[event["Min_Index"], "Og"],
                    color="darkgreen",
                    label="Actual Min Value" if i == first_actual_idx else ""
                )
                plt.axvspan(
                    event["Min_Index"],
                    event["Max_Index"],
                    color="green",
                    alpha=0.3,
                    label="Actual Ramp Event" if i == first_actual_idx else ""
                )

    if not ramps_predicted_plot.empty:
        first_pred_idx = ramps_predicted_plot.index[0]

        for i, event in ramps_predicted_plot.iterrows():
            if event["Max_Index"] in predicted_data_plot.index and event["Min_Index"] in predicted_data_plot.index:
                plt.scatter(
                    event["Max_Index"],
                    predicted_data_plot.loc[event["Max_Index"], "Pred"],
                    color="red",
                    label="Predicted Max Value" if i == first_pred_idx else ""
                )
                plt.scatter(
                    event["Min_Index"],
                    predicted_data_plot.loc[event["Min_Index"], "Pred"],
                    color="green",
                    label="Predicted Min Value" if i == first_pred_idx else ""
                )
                plt.axvspan(
                    event["Min_Index"],
                    event["Max_Index"],
                    color="gray",
                    alpha=0.3,
                    label="Predicted Ramp Event" if i == first_pred_idx else ""
                )

    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.title("Power Signal with Detected Ramps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("Ramp plot saved to:", save_path)