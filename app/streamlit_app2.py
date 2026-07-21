import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wind Power Ramp Forecasting Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    background:#f7f9fc;
}

section[data-testid="stSidebar"]{
    background:#13294B;
}

section[data-testid="stSidebar"] *{
    color:white;
}

.metric-card{
    background:white;
    border-radius:12px;
    padding:18px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

.block-container{
    padding-top:2rem;
}

h1{
    color:#0B3C5D;
}

h2{
    color:#1E5AA8;
}

footer{
    visibility:hidden;
}

#MainMenu{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"

METRICS_DIR = OUTPUT_DIR / "metrics"

PLOTS_DIR = OUTPUT_DIR / "plots"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

DATA_DIR = BASE_DIR / "data" / "processed"

metrics_path = METRICS_DIR / "metrics.json"

ramp_scores_path = METRICS_DIR / "ramp_scores.json"

forecast_plot_path = PLOTS_DIR / "actual_vs_predicted.png"

ramp_plot_path = PLOTS_DIR / "ramp_events_plot.png"

mode_plot_path = PLOTS_DIR / "mode_2_plot.png"

imf1_plot_path = PLOTS_DIR / "imf_prediction_imf1.png"

imf2_plot_path = PLOTS_DIR / "imf_prediction_imf2.png"

imf3_plot_path = PLOTS_DIR / "imf_prediction_imf3.png"

forecast_csv = PREDICTION_DIR / "forecast.csv"

ramps_actual_path = METRICS_DIR / "ramps_actual.csv"

ramps_predicted_path = METRICS_DIR / "ramps_predicted.csv"

# --------------------------------------------------
# Load Files
# --------------------------------------------------

metrics = {}

if metrics_path.exists():

    with open(metrics_path) as f:

        metrics = json.load(f)

ramp_scores = {}

if ramp_scores_path.exists():

    with open(ramp_scores_path) as f:

        ramp_scores = json.load(f)

forecast_df = None

if forecast_csv.exists():

    forecast_df = pd.read_csv(forecast_csv)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🌬 Wind Power")

st.sidebar.caption("Ramp Forecasting Dashboard")

page = st.sidebar.radio(

    "Navigation",

    [

        "Overview",

        "Forecast Analysis",

        "Ramp Detection",

        "IMF Components",

        "Prediction Results",

        "Methodology",

        "Research Findings",

        "About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.subheader("Project")

st.sidebar.write("Forecast Horizon")

st.sidebar.success("8 Hours")

st.sidebar.write("Forecasting")

st.sidebar.info("VMD + XGBoost")

st.sidebar.write("Ramp Detection")

st.sidebar.warning("DSI")

st.sidebar.markdown("---")

st.sidebar.subheader("Technologies")

st.sidebar.write("Python")

st.sidebar.write("XGBoost")

st.sidebar.write("VMD")

st.sidebar.write("Streamlit")

st.sidebar.write("Pandas")

st.sidebar.write("Plotly")

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def metric_box(title, value, delta=None):

    st.metric(title, value, delta)


def section_title(title, subtitle=""):

    st.markdown(f"# {title}")

    if subtitle:

        st.write(subtitle)


def show_image(path, caption):

    if Path(path).exists():

        st.image(str(path), caption=caption, use_container_width=True)

    else:

        st.warning(f"{caption} not available.")


def download_dataframe(df, filename):

    st.download_button(

        "Download CSV",

        df.to_csv(index=False),

        filename,

        mime="text/csv"

    )


def load_csv(path):

    if Path(path).exists():

        return pd.read_csv(path)

    return None

# ==================================================
# OVERVIEW PAGE
# ==================================================

if page == "Overview":

    section_title(
        "🌬️ Wind Power Ramp Forecasting Dashboard",
        "End-to-End Short-Term Wind Power Ramp Forecasting using Variational Mode Decomposition (VMD), XGBoost and Definition-Based Sign Indicator (DSI)"
    )

    st.markdown(
        """
This dashboard demonstrates an end-to-end framework for **8-hour ahead wind power forecasting**
and **wind ramp event detection**.

The workflow combines:

- **Variational Mode Decomposition (VMD)** to decompose the wind power signal.
- **XGBoost regression models** to forecast each decomposed component.
- **Signal reconstruction** to obtain the final wind power forecast.
- **Definition-Based Sign Indicator (DSI)** for accurate wind ramp detection.

The objective is not only to forecast wind power accurately but also to reliably detect
critical ramp events that influence power system stability and renewable energy integration.
"""
    )

    st.divider()

    # ------------------------------------------------
    # Project Summary
    # ------------------------------------------------

    st.subheader("Project Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info(
            """
### Forecast Horizon

**8 Hours**

Short-term prediction
"""
        )

    with c2:
        st.info(
            """
### Signal Decomposition

**VMD**

Three IMF Components
"""
        )

    with c3:
        st.info(
            """
### Forecast Model

**XGBoost**

One model per IMF
"""
        )

    with c4:
        st.info(
            """
### Ramp Detection

**DSI**

2-hour ramp window
"""
        )

    st.divider()

    # ------------------------------------------------
    # KPI Dashboard
    # ------------------------------------------------

    st.header("Performance Summary")

    if metrics and ramp_scores:

        st.subheader("Forecast Performance")

        fc1, fc2, fc3 = st.columns(3)

        fc1.metric(
            "MAE",
            f"{metrics['mae']:.3f}"
        )

        fc2.metric(
            "RMSE",
            f"{metrics['rmse']:.3f}"
        )

        fc3.metric(
            "R²",
            f"{metrics['r2']:.3f}"
        )

        st.subheader("Ramp Detection Performance")

        rc1, rc2, rc3 = st.columns(3)

        rc1.metric(
            "Precision",
            f"{ramp_scores['precision']:.3f}"
        )

        rc2.metric(
            "Recall",
            f"{ramp_scores['recall']:.3f}"
        )

        rc3.metric(
            "F1 Score",
            f"{ramp_scores['f1_score']:.3f}"
        )

        rc4, rc5, rc6 = st.columns(3)

        rc4.metric(
            "Actual Ramp Events",
            ramp_scores["total_actual_ramps"]
        )

        rc5.metric(
            "Predicted Ramp Events",
            ramp_scores["total_predicted_ramps"]
        )

        rc6.metric(
            "Matched Events",
            ramp_scores["matches"]
        )

    else:

        st.warning("Run the forecasting pipeline to generate evaluation metrics.")

    st.divider()

    # ------------------------------------------------
    # Dynamic Interpretation
    # ------------------------------------------------

    st.header("Automatic Model Interpretation")

    if metrics:

        if metrics["r2"] >= 0.70:

            st.success(
                "The forecasting model demonstrates excellent predictive performance. The predicted wind power closely follows the observed trend across the forecast horizon."
            )

        elif metrics["r2"] >= 0.50:

            st.info(
                "The forecasting model captures the overall wind power trend effectively. Most prediction errors occur during periods of rapid wind power fluctuation, which is expected in highly variable renewable energy systems."
            )

        else:

            st.warning(
                "The model captures the overall behaviour of wind power generation but exhibits larger forecasting errors during highly dynamic operating conditions."
            )

    if ramp_scores:

        if ramp_scores["recall"] >= 0.95:

            st.success(
                "Excellent ramp detection performance. Nearly all observed wind ramp events were successfully detected, demonstrating the operational reliability of the proposed framework."
            )

        if ramp_scores["precision"] >= 0.80:

            st.info(
                "The precision indicates that most predicted ramp events correspond to actual ramp occurrences. A conservative prediction strategy is generally preferred in operational power systems, as missing a true ramp event can have greater consequences than issuing a small number of false alarms."
            )

    st.divider()

    # ------------------------------------------------
    # Workflow
    # ------------------------------------------------

    st.header("Forecasting Workflow")

    st.code(
"""
Raw Wind Power Data
        │
        ▼
Data Cleaning & Feature Engineering
        │
        ▼
Variational Mode Decomposition (VMD)
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
IMF1   IMF2   IMF3
 │      │      │
 ▼      ▼      ▼
XGB    XGB    XGB
 │      │      │
 └──────┼──────┘
        ▼
Signal Reconstruction
        ▼
Final 8-Hour Forecast
        ▼
Definition-Based Sign Indicator
        ▼
Wind Ramp Detection
        ▼
Performance Evaluation
""",
        language="text"
    )

    st.divider()

    # ------------------------------------------------
    # Why Ramp Forecasting?
    # ------------------------------------------------

    st.header("Why Wind Ramp Forecasting Matters")

    st.markdown(
        """
Wind power is inherently variable due to changing atmospheric conditions.
Large and rapid changes in power output, commonly referred to as **wind ramps**, can
create significant operational challenges for modern electricity grids.

Reliable wind ramp forecasting enables:

- Improved power system stability
- Better reserve scheduling
- Reduced balancing costs
- Increased renewable energy penetration
- Enhanced operational decision-making

The proposed VMD–XGBoost–DSI framework addresses these challenges by combining signal decomposition,
machine learning, and event-based ramp detection within a unified forecasting pipeline.
"""
    )

    st.divider()

    # ------------------------------------------------
    # Quick Navigation
    # ------------------------------------------------

    st.success(
        """
Use the navigation panel on the left to explore:

📈 Forecast Analysis

⚡ Ramp Detection

🧩 IMF Components

📋 Prediction Results

🔬 Methodology

📄 Research Findings

ℹ️ About the Project
"""
    )

# ==================================================
# FORECAST ANALYSIS
# ==================================================

elif page == "Forecast Analysis":

    section_title(
        "📈 Forecast Analysis",
        "Evaluation of the reconstructed 8-hour wind power forecast."
    )

    st.markdown("""
The forecasting framework reconstructs the final wind power prediction by integrating
the forecasts generated for each decomposed IMF component.

This page evaluates how accurately the reconstructed signal follows the observed wind
power generation over the forecasting horizon.
""")

    st.divider()

    # ------------------------------------------------
    # Forecast Metrics
    # ------------------------------------------------

    st.subheader("Forecast Performance")

    if metrics:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mean Absolute Error (MAE)",
            f"{metrics['mae']:.3f}"
        )

        c2.metric(
            "Root Mean Square Error (RMSE)",
            f"{metrics['rmse']:.3f}"
        )

        c3.metric(
            "Coefficient of Determination (R²)",
            f"{metrics['r2']:.3f}"
        )

    else:

        st.warning("Forecast metrics are unavailable.")

    st.divider()

    # ------------------------------------------------
    # Forecast Plot
    # ------------------------------------------------

    st.subheader("Actual vs Forecasted Wind Power")

    if forecast_df is not None:

        # Detect column names automatically

        cols = [c.lower() for c in forecast_df.columns]

        actual_col = None
        pred_col = None
        time_col = None

        for c in forecast_df.columns:

            cl = c.lower()

            if "actual" in cl or "og" in cl:

                actual_col = c

            elif "forecast" in cl or "pred" in cl:

                pred_col = c

            elif "time" in cl or "date" in cl:

                time_col = c

        if actual_col and pred_col:

            if time_col:

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=forecast_df[time_col],
                        y=forecast_df[actual_col],
                        name="Actual",
                        line=dict(width=2)
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=forecast_df[time_col],
                        y=forecast_df[pred_col],
                        name="Forecast",
                        line=dict(width=2)
                    )
                )

            else:

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        y=forecast_df[actual_col],
                        name="Actual"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        y=forecast_df[pred_col],
                        name="Forecast"
                    )
                )

            fig.update_layout(

                title="Actual vs Forecasted Wind Power",

                xaxis_title="Time",

                yaxis_title="Wind Power",

                hovermode="x unified",

                height=550

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            show_image(
                forecast_plot_path,
                "Actual vs Forecast"
            )

    else:

        show_image(
            forecast_plot_path,
            "Actual vs Forecast"
        )

    st.divider()

    # ------------------------------------------------
    # Forecast Interpretation
    # ------------------------------------------------

    st.subheader("Forecast Interpretation")

    if metrics:

        r2 = metrics["r2"]

        if r2 >= 0.70:

            st.success(
                """
The forecasting model explains a large proportion of the observed variation in wind
power generation. The reconstructed signal closely follows the measured wind power,
indicating strong predictive capability.
"""
            )

        elif r2 >= 0.50:

            st.info(
                """
The reconstructed forecast successfully captures the overall trend and variability
of wind power generation. Larger prediction errors mainly occur during periods of
rapid power fluctuations, which are characteristic of wind ramp events.
"""
            )

        else:

            st.warning(
                """
The model captures the overall trend but exhibits reduced predictive performance
during highly dynamic operating conditions.
"""
            )

    st.divider()

    # ------------------------------------------------
    # Error Analysis
    # ------------------------------------------------

    if forecast_df is not None and actual_col and pred_col:

        st.subheader("Forecast Error Analysis")

        forecast_df["Residual"] = (

            forecast_df[actual_col]

            -

            forecast_df[pred_col]

        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(

                forecast_df,

                x="Residual",

                nbins=40,

                title="Distribution of Forecast Errors"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

        with col2:

            fig = px.scatter(

                forecast_df,

                x=actual_col,

                y="Residual",

                title="Residuals vs Actual Wind Power"

            )

            fig.add_hline(

                y=0,

                line_dash="dash"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

    st.divider()

    # ------------------------------------------------
    # Forecast Statistics
    # ------------------------------------------------

    if forecast_df is not None and actual_col and pred_col:

        st.subheader("Forecast Statistics")

        stats = pd.DataFrame({

            "Metric":[

                "Average Actual",

                "Average Forecast",

                "Maximum Actual",

                "Maximum Forecast",

                "Minimum Actual",

                "Minimum Forecast"

            ],

            "Value":[

                forecast_df[actual_col].mean(),

                forecast_df[pred_col].mean(),

                forecast_df[actual_col].max(),

                forecast_df[pred_col].max(),

                forecast_df[actual_col].min(),

                forecast_df[pred_col].min()

            ]

        })

        st.dataframe(

            stats,

            use_container_width=True

        )

    st.divider()

    # ------------------------------------------------
    # Forecast Data
    # ------------------------------------------------

    st.subheader("Forecast Data")

    if forecast_df is not None:

        st.dataframe(

            forecast_df,

            use_container_width=True,

            height=400

        )

        download_dataframe(

            forecast_df,

            "forecast_results.csv"

        )

    else:

        st.warning(

            "Forecast CSV not found. Run main.py to generate prediction results."

        )

    st.divider()

    st.success(
        """
Forecast analysis completed.

Continue to **Ramp Detection** to evaluate how effectively the proposed framework
identifies significant wind ramp events.
"""
    )

# ==================================================
# RAMP DETECTION
# ==================================================

elif page == "Ramp Detection":

    section_title(
        "⚡ Wind Ramp Detection",
        "Evaluation of the proposed Definition-Based Sign Indicator (DSI) algorithm for detecting significant wind ramp events."
    )

    st.markdown("""
Wind ramps are rapid increases or decreases in wind power over a short duration.
Accurate detection of these events is essential for maintaining power system stability,
optimising reserve allocation, and improving renewable energy integration.

The proposed framework applies the **Definition-Based Sign Indicator (DSI)** algorithm
to identify ramp events from both the observed and forecasted wind power signals.
""")

    st.divider()

    # ------------------------------------------------
    # KPI Dashboard
    # ------------------------------------------------

    st.subheader("Ramp Detection Performance")

    if ramp_scores:

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Precision",
            f"{ramp_scores['precision']:.3f}"
        )

        k2.metric(
            "Recall",
            f"{ramp_scores['recall']:.3f}"
        )

        k3.metric(
            "F1 Score",
            f"{ramp_scores['f1_score']:.3f}"
        )

        st.write("")

        k4, k5, k6 = st.columns(3)

        k4.metric(
            "Actual Ramp Events",
            ramp_scores["total_actual_ramps"]
        )

        k5.metric(
            "Predicted Ramp Events",
            ramp_scores["total_predicted_ramps"]
        )

        k6.metric(
            "Matched Events",
            ramp_scores["matches"]
        )

    else:

        st.warning("Ramp detection results are unavailable.")

    st.divider()

    # ------------------------------------------------
    # Performance Interpretation
    # ------------------------------------------------

    st.subheader("Operational Interpretation")

    if ramp_scores:

        precision = ramp_scores["precision"]
        recall = ramp_scores["recall"]
        f1 = ramp_scores["f1_score"]

        if recall >= 0.95:

            st.success(
                """
### Excellent Ramp Detection Capability

The proposed framework successfully identifies almost every observed wind ramp event.
High recall is particularly valuable in operational environments where missing a
critical ramp may lead to power imbalance and increased reserve costs.
"""
            )

        if precision >= 0.80:

            st.info(
                """
### Reliable Event Prediction

Most predicted ramp events correspond to actual ramps.
A small number of additional predicted ramps represents a conservative operating
strategy, which is generally preferred over missing genuine events.
"""
            )

        if f1 >= 0.85:

            st.success(
                """
### Balanced Performance

The high F1-score indicates an effective balance between detecting true ramp events
and minimising false detections.
"""
            )

    st.divider()

    # ------------------------------------------------
    # Event Distribution
    # ------------------------------------------------

    st.subheader("Detected Ramp Events")

    if ramp_plot_path.exists():

        st.image(
            str(ramp_plot_path),
            caption="Detected Wind Ramp Events",
            use_container_width=True
        )

    else:

        st.warning("Ramp event plot not available.")

    st.divider()

    # ------------------------------------------------
    # Summary Cards
    # ------------------------------------------------

    st.subheader("Event Summary")

    if ramp_scores:

        actual = ramp_scores["total_actual_ramps"]
        predicted = ramp_scores["total_predicted_ramps"]
        matched = ramp_scores["matches"]

        summary = pd.DataFrame({

            "Metric":[

                "Observed Ramp Events",

                "Predicted Ramp Events",

                "Correctly Matched Events",

                "Detection Recall",

                "Prediction Precision",

                "Overall F1 Score"

            ],

            "Value":[

                actual,

                predicted,

                matched,

                recall,

                precision,

                f1

            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )

    st.divider()

    # ------------------------------------------------
    # Actual vs Predicted Tables
    # ------------------------------------------------

    st.subheader("Ramp Event Records")

    actual_df = load_csv(ramps_actual_path)
    predicted_df = load_csv(ramps_predicted_path)

    tab1, tab2 = st.tabs([
        "Observed Ramp Events",
        "Predicted Ramp Events"
    ])

    with tab1:

        if actual_df is not None:

            st.dataframe(
                actual_df,
                use_container_width=True,
                height=350
            )

            download_dataframe(
                actual_df,
                "observed_ramps.csv"
            )

        else:

            st.warning("Observed ramp table not found.")

    with tab2:

        if predicted_df is not None:

            st.dataframe(
                predicted_df,
                use_container_width=True,
                height=350
            )

            download_dataframe(
                predicted_df,
                "predicted_ramps.csv"
            )

        else:

            st.warning("Predicted ramp table not found.")

    st.divider()

    # ------------------------------------------------
    # Research Discussion
    # ------------------------------------------------

    st.subheader("Research Discussion")

    st.markdown("""

The primary objective of the proposed framework is **accurate wind ramp detection**
rather than minimising forecast error alone.

By first decomposing the wind power signal using **Variational Mode Decomposition (VMD)**,
the forecasting model captures both long-term trends and high-frequency fluctuations.
The reconstructed forecast is subsequently analysed using the
**Definition-Based Sign Indicator (DSI)** algorithm to identify significant ramp events.

The reported performance demonstrates that the proposed framework successfully captures
nearly all operationally important wind ramps while maintaining strong overall
prediction reliability.

This makes the framework suitable for applications such as:

- Grid stability monitoring
- Reserve scheduling
- Renewable energy integration
- Power system operational planning
- Wind farm decision support

""")

    st.divider()

    # ------------------------------------------------
    # Operational Assessment
    # ------------------------------------------------

    st.subheader("Operational Assessment")

    if ramp_scores:

        if recall == 1.0:

            st.success(
                """
The framework achieved **100% recall**, indicating that every observed ramp
event was successfully detected. This is particularly valuable because missing
critical wind ramps can adversely affect grid reliability.
"""
            )

        if precision < 1.0:

            st.info(
                """
The model predicts a small number of additional ramp events beyond those observed.
Such conservative behaviour is acceptable in operational forecasting because
false alarms generally have a lower impact than undetected ramp events.
"""
            )

    st.divider()

    st.success(
        """
The proposed VMD–XGBoost–DSI framework demonstrates reliable wind ramp detection,
making it suitable for supporting operational decision-making in modern renewable
energy systems.
"""
    )

# ==================================================
# IMF COMPONENTS
# ==================================================

elif page == "IMF Components":

    section_title(
        "🧩 IMF Component Analysis",
        "Variational Mode Decomposition (VMD) separates the original wind power signal into multiple Intrinsic Mode Functions (IMFs), enabling each frequency component to be modelled independently."
    )

    st.markdown("""
The forecasting framework first decomposes the wind power signal using **Variational Mode Decomposition (VMD)**.
Each Intrinsic Mode Function (IMF) represents a different frequency band of the original signal.

Independent XGBoost regression models are trained for each IMF, and the individual predictions are
combined to reconstruct the final 8-hour wind power forecast.

This decomposition enables the forecasting model to capture both long-term trends and short-term
variations more effectively than modelling the raw signal directly.
""")

    st.divider()

    # ------------------------------------------------
    # Overview
    # ------------------------------------------------

    st.subheader("Why Use VMD?")

    col1, col2 = st.columns([1, 2])

    with col1:

        st.info("""
### Advantages

- Reduces signal complexity

- Separates frequency modes

- Improves model stability

- Handles non-linear behaviour

- Enhances forecasting accuracy
""")

    with col2:

        st.success("""
### Role within the Forecasting Pipeline

Raw Wind Power

↓

Variational Mode Decomposition

↓

IMF1 + IMF2 + IMF3

↓

Independent XGBoost Models

↓

Signal Reconstruction

↓

Final Wind Power Forecast
""")

    st.divider()

    # ------------------------------------------------
    # IMF 1
    # ------------------------------------------------

    st.header("IMF 1 – High Frequency Component")

    c1, c2 = st.columns([2, 1])

    with c1:

        show_image(
            imf1_plot_path,
            "IMF 1 Prediction"
        )

    with c2:

        st.markdown("""
### Description

Represents the **highest-frequency**
oscillations present in the wind power signal.

This component captures:

- rapid fluctuations

- turbulence

- short-term variability

- sudden local changes

It is generally the most difficult IMF
to predict because of its highly dynamic behaviour.
""")

    st.divider()

    # ------------------------------------------------
    # IMF 2
    # ------------------------------------------------

    st.header("IMF 2 – Intermediate Frequency Component")

    c1, c2 = st.columns([2, 1])

    with c1:

        show_image(
            imf2_plot_path,
            "IMF 2 Prediction"
        )

    with c2:

        st.markdown("""
### Description

Represents medium-frequency
behaviour within the wind power signal.

Captures

- local trends

- moderate oscillations

- changing wind conditions

- transition behaviour

This IMF links the rapidly changing
components with the underlying trend.
""")

    st.divider()

    # ------------------------------------------------
    # IMF 3
    # ------------------------------------------------

    st.header("IMF 3 – Low Frequency Component")

    c1, c2 = st.columns([2, 1])

    with c1:

        show_image(
            imf3_plot_path,
            "IMF 3 Prediction"
        )

    with c2:

        st.markdown("""
### Description

Represents the
lowest-frequency behaviour.

Captures

- long-term trend

- seasonal behaviour

- baseline generation

- overall wind characteristics

This IMF is generally smoother
and easier to forecast than higher-frequency
components.
""")

    st.divider()

    # ------------------------------------------------
    # Mode Plot
    # ------------------------------------------------

    st.header("Selected Component View")

    st.markdown("""
The selected mode provides a detailed view of one decomposed frequency component.
Examining an individual IMF helps illustrate how VMD isolates oscillatory behaviour
before forecasting.
""")

    show_image(
        mode_plot_path,
        "Mode 2 Component"
    )

    st.divider()

    # ------------------------------------------------
    # Research Interpretation
    # ------------------------------------------------

    st.header("Component-Level Interpretation")

    st.markdown("""
The decomposition strategy enables each XGBoost model to learn a simpler prediction
task than modelling the complete wind power signal directly.

Instead of learning one highly complex signal, the forecasting problem is divided
into multiple frequency-specific learning tasks.

This improves:

- model convergence

- prediction stability

- representation of local fluctuations

- reconstruction quality

Following prediction, all IMF forecasts are integrated to reconstruct the final
wind power forecast used for ramp detection.
""")

    st.divider()

    # ------------------------------------------------
    # Summary Table
    # ------------------------------------------------

    st.subheader("IMF Summary")

    summary = pd.DataFrame({

        "Component":[
            "IMF 1",
            "IMF 2",
            "IMF 3"
        ],

        "Frequency":[
            "High",
            "Medium",
            "Low"
        ],

        "Primary Behaviour":[
            "Rapid fluctuations",
            "Local oscillations",
            "Long-term trend"
        ],

        "Prediction Difficulty":[
            "High",
            "Medium",
            "Low"
        ]

    })

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.divider()

    st.success("""
The decomposition of the wind power signal into multiple IMFs enables the forecasting
framework to model different frequency behaviours independently before reconstructing
the final wind power forecast. This multi-stage approach forms the foundation of the
proposed VMD–XGBoost forecasting framework.
""")
    
    # ==================================================
# PREDICTION RESULTS
# ==================================================

elif page == "Prediction Results":

    section_title(
        "📋 Prediction Results",
        "Explore the generated forecast values and compare them with the observed wind power."
    )

    st.markdown("""
This section provides access to the reconstructed wind power forecast generated by
the proposed VMD–XGBoost framework.

Users can inspect the forecast, compare it with the observed values,
download prediction results and analyse forecast errors.
""")

    st.divider()

    # ------------------------------------------------
    # Load Forecast
    # ------------------------------------------------

    if forecast_df is None:

        st.warning("""
Forecast CSV not found.

Run **main.py** first to generate:

outputs/predictions/forecast.csv
""")

    else:

        columns = forecast_df.columns.tolist()

        actual_col = None
        pred_col = None
        time_col = None

        for col in columns:

            name = col.lower()

            if "actual" in name or "og" in name:

                actual_col = col

            elif "forecast" in name or "pred" in name:

                pred_col = col

            elif "time" in name or "date" in name:

                time_col = col

        # --------------------------------------------
        # Summary
        # --------------------------------------------

        st.subheader("Forecast Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Predictions",
            len(forecast_df)
        )

        if actual_col:

            c2.metric(
                "Average Actual",
                f"{forecast_df[actual_col].mean():.2f}"
            )

        if pred_col:

            c3.metric(
                "Average Forecast",
                f"{forecast_df[pred_col].mean():.2f}"
            )

        if actual_col and pred_col:

            mae = abs(
                forecast_df[actual_col] -
                forecast_df[pred_col]
            ).mean()

            c4.metric(
                "Average Error",
                f"{mae:.2f}"
            )

        st.divider()

        # --------------------------------------------
        # Search
        # --------------------------------------------

        st.subheader("Search Forecast Records")

        search = st.text_input(
            "Search any value"
        )

        filtered = forecast_df.copy()

        if search:

            filtered = forecast_df[
                forecast_df.astype(str)
                .apply(
                    lambda x: x.str.contains(
                        search,
                        case=False
                    )
                )
                .any(axis=1)
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            height=450
        )

        st.download_button(
            "⬇ Download Forecast CSV",
            filtered.to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv"
        )

        st.divider()

        # --------------------------------------------
        # Forecast Error
        # --------------------------------------------

        if actual_col and pred_col:

            st.subheader("Prediction Error")

            forecast_df["Absolute Error"] = (

                forecast_df[actual_col]

                -

                forecast_df[pred_col]

            ).abs()

            fig = px.line(

                forecast_df,

                y="Absolute Error",

                title="Absolute Forecast Error"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info("""
Large error values generally occur during periods of
rapid wind power fluctuation, where prediction becomes
more challenging.
""")

        st.divider()

        # --------------------------------------------
        # Forecast Distribution
        # --------------------------------------------

        if pred_col:

            st.subheader("Forecast Distribution")

            fig = px.histogram(

                forecast_df,

                x=pred_col,

                nbins=35,

                title="Distribution of Forecasted Wind Power"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # --------------------------------------------
        # Scatter Plot
        # --------------------------------------------

        if actual_col and pred_col:

            st.subheader("Actual vs Forecast Correlation")

            fig = px.scatter(

                forecast_df,

                x=actual_col,

                y=pred_col,

                opacity=0.6,

                trendline="ols"

            )

            fig.update_layout(
                height=600
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # --------------------------------------------
        # Statistics
        # --------------------------------------------

        st.subheader("Descriptive Statistics")

        st.dataframe(

            forecast_df.describe(),

            use_container_width=True

        )

        st.divider()

        # --------------------------------------------
        # Interpretation
        # --------------------------------------------

        st.subheader("Prediction Insights")

        st.markdown("""

The reconstructed forecast demonstrates the capability of the
proposed VMD–XGBoost framework to model both long-term trends
and short-term fluctuations in wind power generation.

The forecast values shown above represent the integrated
prediction obtained after combining the independently predicted
IMF components.

Users may download these results for further analysis,
validation or operational decision-making.

""")

        st.success("""
Prediction results successfully loaded.

Proceed to **Methodology** to understand the complete forecasting framework.
""")
        
# ==================================================
# METHODOLOGY
# ==================================================

elif page == "Methodology":

    section_title(
        "🔬 Methodology",
        "Overview of the proposed VMD–XGBoost framework for 8-hour wind power forecasting and ramp event detection."
    )

    st.markdown("""
The proposed framework combines **Variational Mode Decomposition (VMD)**,
**Extreme Gradient Boosting (XGBoost)** and a
**Definition-Based Sign Indicator (DSI)** algorithm to accurately forecast
wind power and detect significant wind ramp events.

Unlike conventional forecasting approaches that model the raw signal directly,
the proposed methodology first decomposes the wind power signal into multiple
frequency components before forecasting each component independently.
""")

    st.divider()

    # ------------------------------------------------
    # Pipeline Workflow
    # ------------------------------------------------

    st.subheader("Overall Framework")

    st.info("""

Historical Wind Power Data

↓

Data Pre-processing

↓

Variational Mode Decomposition (VMD)

↓

IMF 1     IMF 2     IMF 3

↓

Independent XGBoost Models

↓

IMF Predictions

↓

Signal Reconstruction

↓

8-Hour Wind Power Forecast

↓

Definition-Based Sign Indicator (DSI)

↓

Wind Ramp Detection

""")

    st.divider()

    # ------------------------------------------------
    # Step 1
    # ------------------------------------------------

    st.header("Step 1 — Data Pre-processing")

    st.markdown("""

The raw wind power dataset undergoes several pre-processing steps before model
development.

These include:

- Handling missing observations
- Data cleaning
- Time-order preservation
- Feature preparation
- Train-test splitting

Proper pre-processing ensures the forecasting model receives clean and consistent
input data while preventing information leakage.

""")

    st.divider()

    # ------------------------------------------------
    # Step 2
    # ------------------------------------------------

    st.header("Step 2 — Variational Mode Decomposition (VMD)")

    st.markdown("""

Variational Mode Decomposition separates the original wind power signal into
multiple Intrinsic Mode Functions (IMFs).

Each IMF captures a different frequency characteristic:

• IMF 1 — High-frequency fluctuations

• IMF 2 — Medium-frequency oscillations

• IMF 3 — Long-term trend

Instead of learning one highly complex signal,
the forecasting problem is divided into multiple simpler prediction tasks.

""")

    st.success("""
Advantages of VMD

• Reduces signal complexity

• Minimises mode mixing

• Preserves frequency information

• Improves prediction stability

• Enables better modelling of non-linear behaviour
""")

    st.divider()

    # ------------------------------------------------
    # Step 3
    # ------------------------------------------------

    st.header("Step 3 — XGBoost Forecasting")

    st.markdown("""

Each decomposed IMF is modelled independently using an XGBoost regression model.

Separate learning enables each model to specialise in the behaviour of a
particular frequency component.

After prediction, the IMF forecasts are combined to reconstruct the final
wind power forecast.

""")

    st.divider()

    # ------------------------------------------------
    # Step 4
    # ------------------------------------------------

    st.header("Step 4 — Signal Reconstruction")

    st.markdown("""

The predicted IMF components are summed together to reconstruct the complete
8-hour wind power forecast.

This reconstructed signal is subsequently evaluated using forecasting metrics
including:

• Mean Absolute Error (MAE)

• Root Mean Square Error (RMSE)

• Coefficient of Determination (R²)

""")

    st.divider()

    # ------------------------------------------------
    # Step 5
    # ------------------------------------------------

    st.header("Step 5 — Wind Ramp Detection")

    st.markdown("""

Following signal reconstruction, the forecast is analysed using the
Definition-Based Sign Indicator (DSI) algorithm.

The DSI algorithm identifies periods where wind power changes exceed a predefined
threshold within a specified duration.

Detected events are compared against observed ramps to evaluate operational
performance.

Performance is measured using:

• Precision

• Recall

• F1 Score

""")

    st.divider()

    # ------------------------------------------------
    # Why This Approach
    # ------------------------------------------------

    st.header("Why This Framework?")

    st.markdown("""

Traditional forecasting models often struggle with rapidly changing wind power
because they attempt to learn the complete signal directly.

The proposed framework overcomes this challenge by:

- decomposing the signal,
- modelling each frequency independently,
- reconstructing the forecast,
- detecting operationally significant ramp events.

This improves both interpretability and forecasting robustness.

""")

    st.divider()

    # ------------------------------------------------
    # Research Contribution
    # ------------------------------------------------

    st.header("Research Contribution")

    st.success("""

The primary contribution of this work is the integration of:

• Variational Mode Decomposition (VMD)

• XGBoost Regression

• Definition-Based Sign Indicator (DSI)

into a unified framework for
8-hour wind power forecasting and wind ramp detection.

The methodology improves the representation of complex wind dynamics while
maintaining strong ramp detection capability for power system applications.

""")

    st.divider()

    # ------------------------------------------------
    # Key Takeaways
    # ------------------------------------------------

    st.subheader("Key Takeaways")

    summary = pd.DataFrame({

        "Stage":[
            "Pre-processing",
            "VMD",
            "XGBoost",
            "Integration",
            "DSI"
        ],

        "Purpose":[
            "Prepare clean dataset",
            "Extract frequency components",
            "Forecast each IMF",
            "Reconstruct wind power",
            "Detect ramp events"
        ]

    })

    st.dataframe(
        summary,
        use_container_width=True
    )

    st.success("""
The proposed methodology provides an interpretable and effective framework for
wind power forecasting while maintaining excellent wind ramp detection capability,
making it suitable for renewable energy forecasting and smart grid applications.
""")
    
    # ==================================================
# ABOUT THE PROJECT
# ==================================================

elif page == "About":

    section_title(
        "👨‍💻 About the Project",
        "Research-driven wind power forecasting and ramp detection framework."
    )

    st.markdown("""
This dashboard accompanies a research project focused on **8-hour wind power forecasting**
and **wind ramp event detection** using a hybrid machine learning framework.

The project integrates signal decomposition, ensemble machine learning and
event-based analysis to improve renewable energy forecasting and support
power system operation.
""")

    st.divider()

    # ------------------------------------------------
    # Author Information
    # ------------------------------------------------

    st.header("Author")

    col1, col2 = st.columns([1,3])

    with col1:

        st.markdown("## 👤")

    with col2:

        st.markdown("""
### Leechita Gopalakrishnan

**Master of Data Science and Innovation**

University of Technology Sydney (UTS)

Research Area:

- Wind Power Forecasting
- Wind Ramp Detection
- Machine Learning
- Renewable Energy Analytics
- Time Series Forecasting
""")

    st.divider()

    # ------------------------------------------------
    # Research Overview
    # ------------------------------------------------

    st.header("Research Overview")

    st.markdown("""
This research investigates the use of **Variational Mode Decomposition (VMD)**,
**Extreme Gradient Boosting (XGBoost)** and a
**Definition-Based Sign Indicator (DSI)** algorithm for short-term wind power
forecasting and operational wind ramp detection.

The framework decomposes the original wind power signal into multiple intrinsic
mode functions (IMFs), forecasts each component independently and reconstructs
the final wind power prediction before identifying significant ramp events.

The primary objective is not only to improve forecast quality but also to maximise
the reliability of wind ramp detection, which is essential for modern power
system operation.
""")

    st.divider()

    # ------------------------------------------------
    # Technologies
    # ------------------------------------------------

    st.header("Technologies Used")

    tech = pd.DataFrame({

        "Technology":[
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-Learn",
            "XGBoost",
            "Variational Mode Decomposition",
            "Streamlit",
            "Plotly",
            "Matplotlib"
        ],

        "Purpose":[
            "Programming",
            "Data Processing",
            "Scientific Computing",
            "Machine Learning",
            "Regression Model",
            "Signal Decomposition",
            "Interactive Dashboard",
            "Interactive Visualisation",
            "Static Visualisation"
        ]

    })

    st.dataframe(
        tech,
        use_container_width=True
    )

    st.divider()

    # ------------------------------------------------
    # Framework Summary
    # ------------------------------------------------

    st.header("Framework Summary")

    st.info("""

Historical Wind Power

↓

Data Cleaning

↓

Variational Mode Decomposition

↓

IMF Forecasting

↓

Signal Reconstruction

↓

8-Hour Wind Power Forecast

↓

Definition-Based Sign Indicator

↓

Wind Ramp Detection

""")

    st.divider()

    # ------------------------------------------------
    # Key Features
    # ------------------------------------------------

    st.header("Key Features")

    st.success("""

✔ 8-Hour Wind Power Forecasting

✔ Variational Mode Decomposition

✔ IMF-based Forecasting

✔ XGBoost Regression

✔ Signal Reconstruction

✔ Wind Ramp Detection

✔ Interactive Dashboard

✔ Forecast Evaluation

✔ Ramp Detection Evaluation

✔ Downloadable Results

""")

    st.divider()

    # ------------------------------------------------
    # Research Impact
    # ------------------------------------------------

    st.header("Potential Applications")

    st.markdown("""

The proposed framework can support:

- Renewable Energy Forecasting

- Smart Grid Operation

- Wind Farm Monitoring

- Grid Stability Assessment

- Reserve Scheduling

- Energy Market Planning

- Power System Decision Support

""")

    st.divider()

# ------------------------------------------------
# Publication
# ------------------------------------------------

st.header("📄 Associated Publication")

st.markdown("""
This dashboard accompanies the implementation presented in the following research publication.

The proposed framework combines:

- Variational Mode Decomposition (VMD)
- XGBoost Regression
- Definition-Based Sign Indicator (DSI)

to perform **8-hour wind power forecasting** and **wind ramp detection** for renewable energy applications.
""")

st.info("""
📚 **Citation**

Authors: Leechita Gopalakrishnan et al.

Title: Short-term Wind Power Ramp Forecasting Using Sequential Approach

Journal/Conference: 2024 International Conference on Modeling, Simulation & Intelligent Computing (MoSICom), Dubai, United Arab Emirates, 2024

Year: 2024

DOI: 10.1109/MoSICom63082.2024.10881012.
""")

st.divider()

    # ------------------------------------------------
    # Repository
    # ------------------------------------------------

st.header("Repository")

st.markdown("""
The GitHub repository contains:

- Source code
- Data processing pipeline
- Forecasting models
- Streamlit dashboard
- Visualisations
- Documentation
- Prediction outputs
""")

st.divider()

# ------------------------------------------------
# Acknowledgements
# ------------------------------------------------

st.header("Acknowledgements")

st.markdown("""
This research project was developed as part of Mitacs Globalink Research internship 
under Prof. Dr. Julian L Cardenas Barrera from University of New Brunswick, Canada
under machine learning and renewable energy analytics.

Special thanks to supervisors, researchers and the open-source
Python community for providing the tools that made this work possible.
""")

st.divider()
# ------------------------------------------------
# Footer
# ------------------------------------------------

st.markdown("---")

st.markdown(
        """
<div style='text-align:center;'>

### Wind Power Ramp Forecasting Dashboard

Developed by **Leechita Gopalakrishnan**

University of Technology Sydney

© 2026

</div>
""",unsafe_allow_html=True)