"""Shared helpers for loading the generated dataset and building a live facility snapshot."""
from pathlib import Path
import pandas as pd
from planner.facility_config import ZONES
from app.database import get_equipment_status, get_active_opt_outs

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"


def load_dataset():
    load_df = pd.read_csv(DATA_DIR / "load_timeseries.csv", parse_dates=["timestamp"])
    occ_df = pd.read_csv(DATA_DIR / "occupancy.csv", parse_dates=["timestamp"])
    return load_df, occ_df


def current_snapshot(load_df: pd.DataFrame, occ_df: pd.DataFrame, index: int = None):
    """Pick a representative interval to demo (defaults to the day's highest-load interval)."""
    if index is None:
        index = int(load_df["total_load_kw"].idxmax())
    row = load_df.iloc[index]
    ts = row["timestamp"]
    occ_at_ts = occ_df[occ_df["timestamp"] == ts]

    zone_temps = {}
    for z in ZONES:
        col = f"temp_{z.zone_id}"
        if col in row and pd.notna(row[col]):
            zone_temps[z.zone_id] = float(row[col])
        else:
            m = occ_at_ts[occ_at_ts["zone_id"] == z.zone_id]
            zone_temps[z.zone_id] = float(m["temperature_c"].iloc[0]) if len(m) else None

    occ_band = "UNKNOWN"
    if len(occ_at_ts):
        bands = occ_at_ts["occupancy_band"].dropna()
        if len(bands):
            order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
            occ_band = max(bands, key=lambda b: order.get(b, 0))

    return {
        "timestamp": ts,
        "current_load_kw": float(row["total_load_kw"]),
        "production_load_kw": float(row["production_load_kw"]),
        "zone_temperatures": zone_temps,
        "occupancy_band": occ_band,
        "occupancy_known": bool(len(occ_at_ts["occupancy_band"].dropna())) if len(occ_at_ts) else False,
        "scenario": row.get("scenario", "NORMAL"),
    }


def live_facility_state():
    return {
        "equipment_status": get_equipment_status(),
        "opted_out_zones": set(get_active_opt_outs()),
    }
