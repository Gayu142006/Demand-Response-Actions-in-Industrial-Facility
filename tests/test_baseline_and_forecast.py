import pandas as pd
import numpy as np
from planner.baseline import demand_charge, energy_cost, compute_baseline
from planner.demand_forecast import (
    add_time_features, historical_average_forecast, train_test_split_time,
    ml_forecast_train_predict, forecast_errors,
)
from planner.facility_config import DEMAND_RATE_PER_KW


def make_fake_day(n=96, peak=824.0):
    ts = pd.date_range("2026-06-01", periods=n, freq="15min")
    load = np.linspace(400, peak, n // 2).tolist() + np.linspace(peak, 400, n - n // 2).tolist()
    tariff = [8.5] * n
    return pd.DataFrame({"timestamp": ts, "total_load_kw": load, "tariff_per_kwh": tariff})


def test_demand_charge_scales_linearly():
    assert demand_charge(100) == 100 * DEMAND_RATE_PER_KW
    assert demand_charge(0) == 0


def test_energy_cost_positive():
    df = make_fake_day()
    cost = energy_cost(df)
    assert cost > 0


def test_compute_baseline_peak_matches_max():
    df = make_fake_day(peak=900)
    occ = pd.DataFrame({"timestamp": df["timestamp"], "zone_id": ["OFFICE_A"] * len(df),
                          "temperature_c": [22.0] * len(df)})
    result = compute_baseline(df, occ)
    assert result.peak_kw == 900.0
    assert result.demand_charge == 900.0 * DEMAND_RATE_PER_KW


def test_historical_average_forecast_runs():
    df = make_fake_day()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    preds = historical_average_forecast(df)
    assert len(preds) == len(df)
    assert not preds.isna().any()


def test_ml_forecast_reasonable_error():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-06-01", periods=96 * 20, freq="15min")
    hour = ts.hour + ts.minute / 60
    load = 500 + 200 * np.sin((hour / 24) * 2 * np.pi) + rng.normal(0, 10, len(ts))
    df = pd.DataFrame({"timestamp": ts, "total_load_kw": load, "occupancy": 0.5})
    train, test = train_test_split_time(df, test_frac=0.2)
    preds, _ = ml_forecast_train_predict(train, test)
    errs = forecast_errors(test["total_load_kw"].values, preds)
    assert errs["mae"] < 100  # sanity bound, not a hand-picked target
