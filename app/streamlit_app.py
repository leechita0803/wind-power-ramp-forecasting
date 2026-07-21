import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Wind Power Forecasting Dashboard", layout="wide")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
DATA_DIR = BASE_DIR / "data" / "processed"

metrics_path = METRICS_DIR / "metrics.json"
ramp_scores_path = METRICS_DIR / "ramp_scores.json"
ramps_actual_path = METRICS_DIR / "ramps_actual.csv"
ramps_predicted_path = METRICS_DIR / "ramps_predicted.csv"

forecast_plot_path = PLOTS_DIR / "actual_vs_predicted.png"
ramp_plot_path = PLOTS_DIR / "ramp_events_plot.png"
mode_plot_path = PLOTS_DIR / "mode_2_plot.png"

imf1_plot_path = PLOTS_DIR / "imf_prediction_imf1.png"
imf2_plot_path = PLOTS_DIR / "imf_prediction_imf2.png"
imf3_plot_path = PLOTS_DIR / "imf_prediction_imf3.png"

# Optional prediction table file if you save it later
pred_table_path = DATA_DIR / "forecast_comparison.csv"

# -----------------------------
# Header
# -----------------------------
st.title("Wind Power Forecasting Dashboard")
st.write(
    """
This dashboard presents the end-to-end wind power forecasting results.
It shows how well the model predicts wind power, how the signal was broken into components,
and how ramp events were detected and evaluated.

Use this page to understand:
- how accurate the forecast is
- how actual and predicted wind power compare
- whether important ramp events were captured
- how each IMF/component behaves
"""
)

# -----------------------------
# Forecast evaluation
# -----------------------------
st.header("1. Forecast Performance Summary")
st.write(
    """
These metrics summarise the quality of the final wind power forecast.

- **MAE** shows the average absolute error.
- **RMSE** gives more penalty to larger errors.
- **R²** shows how much of the variation in wind power is explained by the model.
"""
)

if metrics_path.exists():
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:.3f}")
    c2.metric("RMSE", f"{metrics['rmse']:.3f}")
    c3.metric("R²", f"{metrics['r2']:.3f}")
else:
    st.warning("Forecast metrics are not available yet. Run the pipeline first.")

# -----------------------------
# Ramp detection summary
# -----------------------------
st.header("2. Ramp Detection Summary")
st.write(
    """
Ramp events are important sudden changes in wind power over a given time window.
This section shows how well the predicted ramps matched the actual ramps.

- **Precision**: how many predicted ramps were correct
- **Recall**: how many actual ramps were successfully detected
- **F1 Score**: balance between precision and recall
"""
)

if ramp_scores_path.exists():
    with open(ramp_scores_path, "r") as f:
        ramp_scores = json.load(f)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{ramp_scores['precision']:.3f}")
    c2.metric("Recall", f"{ramp_scores['recall']:.3f}")
    c3.metric("F1 Score", f"{ramp_scores['f1_score']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Actual Ramp Events", ramp_scores["total_actual_ramps"])
    c5.metric("Predicted Ramp Events", ramp_scores["total_predicted_ramps"])
    c6.metric("Matched Events", ramp_scores["matches"])
else:
    st.warning("Ramp detection scores are not available yet.")

# -----------------------------
# Main forecast plot
# -----------------------------
st.header("3. Actual vs Forecasted Wind Power")
st.write(
    """
This plot compares the real wind power output with the forecasted output.
A closer overlap between the two lines indicates better model performance.
"""
)

if forecast_plot_path.exists():
    st.image(str(forecast_plot_path), caption="Actual vs Forecasted Wind Power", use_column_width=True)
else:
    st.warning("Forecast comparison plot not found.")

# -----------------------------
# Actual and forecasted values table
# -----------------------------
st.header("4. Actual and Forecasted Wind Power Values")
st.write(
    """
This table provides the underlying values behind the forecast.
It helps the client inspect the real and predicted wind power outputs directly.
"""
)

forecast_df = None

if pred_table_path.exists():
    forecast_df = pd.read_csv(pred_table_path)
else:
    # Try to reconstruct a useful table from available files only if you later save one.
    # For now, show guidance if file is absent.
    forecast_df = None

if forecast_df is not None:
    st.dataframe(forecast_df, use_container_width=True)
else:
    forecast_file = Path("outputs/predictions/forecast.csv")

    if forecast_file.exists():

        comparison_df = pd.read_csv(forecast_file)

        st.dataframe(
            comparison_df,
            use_container_width=True
            )

        st.download_button(
            "Download Forecast CSV",
            comparison_df.to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv"
            )

    else:

        st.warning("Run main.py first to generate forecast results.")

# -----------------------------
# Ramp event plot
# -----------------------------
st.header("5. Ramp Event Visualisation")
st.write(
    """
This plot highlights the ramp events detected in both actual and predicted wind power.
It helps assess whether the forecast captured the timing and magnitude of rapid power changes.
"""
)

if ramp_plot_path.exists():
    st.image(str(ramp_plot_path), caption="Detected Ramp Events in Actual and Forecasted Power", use_column_width=True)
else:
    st.warning("Ramp events plot not found.")

# -----------------------------
# IMF/component plots
# -----------------------------
st.header("6. IMF Prediction Components")
st.write(
    """
The forecasting approach models separate IMF components and then combines them into the final prediction.
These plots show the predicted behaviour of each IMF/component.
"""
)

c1, c2, c3 = st.columns(3)

with c1:
    if imf1_plot_path.exists():
        st.image(str(imf1_plot_path), caption="IMF 1 Prediction", use_column_width=True)
    else:
        st.warning("IMF 1 plot not found.")

with c2:
    if imf2_plot_path.exists():
        st.image(str(imf2_plot_path), caption="IMF 2 Prediction", use_column_width=True)
    else:
        st.warning("IMF 2 plot not found.")

with c3:
    if imf3_plot_path.exists():
        st.image(str(imf3_plot_path), caption="IMF 3 Prediction", use_column_width=True)
    else:
        st.warning("IMF 3 plot not found.")

# -----------------------------
# Mode plot
# -----------------------------
st.header("7. Selected Component View")
st.write(
    """
This plot shows one selected signal mode/component from the decomposition process.
It gives a more detailed view of component-level behaviour.
"""
)

if mode_plot_path.exists():
    st.image(str(mode_plot_path), caption="Mode 2 Component", use_column_width=True)
else:
    st.warning("Mode 2 plot not found.")

# -----------------------------
# Ramp event tables
# -----------------------------
st.header("8. Detected Ramp Event Details")
st.write(
    """
These tables list the detected ramp events for both actual and predicted wind power.
They include event timing, magnitude, and direction.
"""
)

tab1, tab2 = st.tabs(["Actual Ramp Events", "Predicted Ramp Events"])

with tab1:
    if ramps_actual_path.exists():
        ramps_actual_df = pd.read_csv(ramps_actual_path)
        st.dataframe(ramps_actual_df, use_container_width=True)
    else:
        st.warning("Actual ramp events table not found.")

with tab2:
    if ramps_predicted_path.exists():
        ramps_predicted_df = pd.read_csv(ramps_predicted_path)
        st.dataframe(ramps_predicted_df, use_container_width=True)
    else:
        st.warning("Predicted ramp events table not found.")