"""
Feature engineering: merge NDVI + weather + mandi data into
a model-ready DataFrame with lag features, rolling statistics,
and seasonal dummies.
"""

import pandas as pd
import numpy as np


def build_features(ndvi_df: pd.DataFrame,
                   weather_df: pd.DataFrame,
                   mandi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge three data sources on date and compute derived features.
    
    Returns a DataFrame with columns:
        date, price, ndvi, rainfall, temp_max, temp_min, humidity,
        lag_1, lag_7, lag_14, roll_7_mean, roll_14_mean, roll_7_std,
        price_momentum, ndvi_change, season_spring, season_summer,
        season_kharif, season_rabi, arrivals_q
    """
    # ── 1. Normalize dates ────────────────────────────────────────────────────
    for df in [ndvi_df, weather_df, mandi_df]:
        df["date"] = pd.to_datetime(df["date"])

    # ── 2. Merge on date ──────────────────────────────────────────────────────
    df = mandi_df[["date", "modal_price", "arrivals_q"]].copy()
    df = df.rename(columns={"modal_price": "price"})
    df = df.merge(ndvi_df[["date", "ndvi"]], on="date", how="left")
    df = df.merge(
        weather_df[["date", "rainfall", "temp_max", "temp_min", "humidity"]],
        on="date", how="left",
    )
    df = df.sort_values("date").reset_index(drop=True)

    # ── 3. Lag features ───────────────────────────────────────────────────────
    for lag in [1, 3, 7, 14]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    # ── 4. Rolling statistics ─────────────────────────────────────────────────
    df["roll_7_mean"]  = df["price"].rolling(7,  min_periods=1).mean()
    df["roll_14_mean"] = df["price"].rolling(14, min_periods=1).mean()
    df["roll_30_mean"] = df["price"].rolling(30, min_periods=1).mean()
    df["roll_7_std"]   = df["price"].rolling(7,  min_periods=1).std().fillna(0)

    # ── 5. Derived signals ────────────────────────────────────────────────────
    df["price_momentum"] = df["price"] / (df["roll_7_mean"] + 1e-6) - 1
    df["ndvi_change"]    = df["ndvi"].diff().fillna(0)
    df["rain_7d_sum"]    = df["rainfall"].rolling(7, min_periods=1).sum()
    df["arrivals_norm"]  = (df["arrivals_q"] - df["arrivals_q"].mean()) / (df["arrivals_q"].std() + 1e-6)
    df["temp_range"]     = df["temp_max"] - df["temp_min"]

    # ── 6. Seasonal dummies (Indian crop calendar) ────────────────────────────
    df["month"]          = df["date"].dt.month
    df["season_rabi"]    = df["month"].isin([11, 12, 1, 2, 3]).astype(int)   # Nov–Mar
    df["season_kharif"]  = df["month"].isin([6, 7, 8, 9, 10]).astype(int)   # Jun–Oct
    df["season_zaid"]    = df["month"].isin([3, 4, 5]).astype(int)           # Mar–May
    df["dow"]            = df["date"].dt.dayofweek   # mandi closures on holidays
    df["day_of_year"]    = df["date"].dt.dayofyear
    # Harvest cycle proxy: cosine of day-of-year
    df["harvest_cycle"]  = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # ── 7. Drop rows with insufficient lag history ────────────────────────────
    df = df.dropna(subset=["lag_7"]).reset_index(drop=True)

    return df


# Feature columns used by the ML model (must match training)
FEATURE_COLS = [
    "ndvi", "ndvi_change",
    "rainfall", "rain_7d_sum", "temp_max", "temp_min", "humidity", "temp_range",
    "lag_1", "lag_3", "lag_7", "lag_14",
    "roll_7_mean", "roll_14_mean", "roll_30_mean", "roll_7_std",
    "price_momentum", "arrivals_norm",
    "season_rabi", "season_kharif", "season_zaid",
    "harvest_cycle", "dow",
]
