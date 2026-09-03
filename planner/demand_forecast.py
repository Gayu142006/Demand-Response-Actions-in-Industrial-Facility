"""
Demand forecasting (spec section 20).

Two forecasters are provided:
  - historical_average_forecast: predicted_load = historical_same_period_average
  - ml_forecast: RandomForestRegressor over time/occupancy/temperature features

Both are evaluated with MAE / RMSE / peak error so the ML model's value-add
(or lack of it) is measured honestly rather than assumed.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def historical_average_forecast(df: pd.DataFrame, target_col="total_load_kw") -> pd.Series:
    """predicted_load = historical_same_period_average, grouped by (day_of_week, hour, minute)."""
    d = add_time_features(df)
    key = ["day_of_week", "hour", d["timestamp"].dt.minute]
    grp_means = d.groupby(["day_of_week", "hour", d["timestamp"].dt.minute.rename("minute")])[target_col].transform("mean")
    return grp_means


def train_test_split_time(df: pd.DataFrame, test_frac=0.2):
    d = df.sort_values("timestamp").reset_index(drop=True)
    n = len(d)
    split = int(n * (1 - test_frac))
    return d.iloc[:split].copy(), d.iloc[split:].copy()


FEATURE_COLS = ["hour", "minute_of_day", "day_of_week", "is_weekend"]


def ml_forecast_train_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col="total_load_kw"):
    train_f = add_time_features(train_df)
    test_f = add_time_features(test_df)

    # occupancy may have missing values (MISSING_DATA scenario) -- impute conservatively
    for d in (train_f, test_f):
        if "occupancy" in d.columns:
            d["occupancy_filled"] = d["occupancy"].fillna(d["occupancy"].median())
        else:
            d["occupancy_filled"] = 0.5

    feats = FEATURE_COLS + ["occupancy_filled"]
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(train_f[feats], train_f[target_col])
    preds = model.predict(test_f[feats])
    return preds, model


def forecast_errors(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    actual_peak = float(np.max(actual))
    predicted_peak = float(np.max(predicted))
    peak_error = abs(actual_peak - predicted_peak)
    peak_error_pct = peak_error / actual_peak * 100 if actual_peak else 0.0
    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "actual_peak_kw": round(actual_peak, 1),
        "predicted_peak_kw": round(predicted_peak, 1),
        "peak_error_kw": round(peak_error, 1),
        "peak_error_pct": round(peak_error_pct, 2),
    }
