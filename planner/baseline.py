"""
Baseline: what would happen if no demand-response action were taken
(spec section 21). Also implements the tariff model (section 13).
"""
from dataclasses import dataclass
import pandas as pd
from planner.facility_config import PEAK_THRESHOLD_KW, DEMAND_RATE_PER_KW


@dataclass
class BaselineResult:
    peak_kw: float
    demand_charge: float
    energy_cost: float
    total_cost: float
    comfort_violation_minutes: int


def demand_charge(peak_kw: float, demand_rate: float = DEMAND_RATE_PER_KW) -> float:
    return peak_kw * demand_rate


def energy_cost(df: pd.DataFrame, interval_hours: float = 0.25) -> float:
    """df must have total_load_kw and tariff_per_kwh columns."""
    kwh = df["total_load_kw"] * interval_hours
    return float((kwh * df["tariff_per_kwh"]).sum())


def comfort_violation_minutes(df_occ_temp: pd.DataFrame, zones_config) -> int:
    """
    Count minutes where a zone's actual temperature exceeds its comfort range.
    In the no-intervention baseline this is normally 0 (no HVAC was touched),
    but we still compute it honestly rather than hard-coding 0.
    """
    from planner.facility_config import get_zone
    minutes = 0
    for _, row in df_occ_temp.iterrows():
        if pd.isna(row.get("temperature_c")):
            continue
        try:
            z = get_zone(row["zone_id"])
        except KeyError:
            continue
        if row["temperature_c"] > z.comfort_max or row["temperature_c"] < z.comfort_min:
            minutes += 15
    return minutes


def compute_baseline(df_day_load: pd.DataFrame, df_day_occ: pd.DataFrame) -> BaselineResult:
    peak = float(df_day_load["total_load_kw"].max())
    dchg = demand_charge(peak)  # demand charge applies to the peak kW for the period, per spec section 13
    ecost = energy_cost(df_day_load)
    violation_min = comfort_violation_minutes(df_day_occ, None)
    return BaselineResult(
        peak_kw=round(peak, 1),
        demand_charge=round(dchg, 2),
        energy_cost=round(ecost, 2),
        total_cost=round(dchg + ecost, 2),
        comfort_violation_minutes=violation_min,
    )
