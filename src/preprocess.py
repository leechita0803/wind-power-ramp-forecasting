import pandas as pd
import numpy as np
from src.config import DATA_PATH, DATETIME_COL, TARGET_COL, TRAIN_RATIO, RESAMPLE_FREQ

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    # Convert datetime column
    df[DATETIME_COL] = pd.to_datetime(
        df[DATETIME_COL],
        dayfirst=True,
        format="mixed",
        errors="coerce"
    )

    # Drop rows where datetime could not be parsed
    df = df.dropna(subset=[DATETIME_COL])

    # Sort and set datetime as index
    df = df.sort_values(DATETIME_COL)
    df = df.set_index(DATETIME_COL)

    # Create time features
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month

    # Convert wind direction into sin/cos
    if "Wind Direction (°)" in df.columns:
        radians = np.deg2rad(df["Wind Direction (°)"])
        df["wind_dir_sin"] = np.sin(radians)
        df["wind_dir_cos"] = np.cos(radians)

    # Drop theoretical power curve if present
    if "Theoretical_Power_Curve (KWh)" in df.columns:
        df = df.drop(columns=["Theoretical_Power_Curve (KWh)"])

    # Keep only non-negative target rows
    df = df[df[TARGET_COL] >= 0]

    # Resample and interpolate
    df = df.resample(RESAMPLE_FREQ).mean()
    df = df.interpolate()

    return df

def split_data(df):
    split_index = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df

def load_and_preprocess_data():
    df = load_data()
    df = preprocess_data(df)
    return df