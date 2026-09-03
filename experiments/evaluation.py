"""
Evaluation experiment (spec sections 40-47).

Runs, for every day in the synthetic dataset:
  A. Baseline (no demand-response action)
  B. Cost-first planner
  C. Occupant-first planner

against the SAME simulated world (same generated load/occupancy/equipment
data), and reports genuine measured metrics -- peak demand, peak reduction %,
demand charge, energy cost, comfort-violation minutes, hard/opt-out
violations, and forecast error (historical-average vs RandomForest).

No numbers here are hand-picked; everything is computed by actually running
the planner and forecaster against the generated dataset.
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from planner.facility_config import PEAK_THRESHOLD_KW, ZONES
from planner.baseline import compute_baseline, demand_charge, energy_cost
from planner.planner_engine import run_planner
from planner.demand_forecast import (
    train_test_split_time, historical_average_forecast, ml_forecast_train_predict, forecast_errors,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_zone_temperatures(day_load_row, day_occ_at_ts):
    zone_temps = {}
    for z in ZONES:
        col = f"temp_{z.zone_id}"
        if col in day_load_row and pd.notna(day_load_row[col]):
            zone_temps[z.zone_id] = float(day_load_row[col])
        else:
            m = day_occ_at_ts[day_occ_at_ts["zone_id"] == z.zone_id]
            zone_temps[z.zone_id] = float(m["temperature_c"].iloc[0]) if len(m) and pd.notna(m["temperature_c"].iloc[0]) else None
    return zone_temps


def simulate_day_with_planner(day_load: pd.DataFrame, day_occ: pd.DataFrame, objective: str):
    """
    Plans a demand-response action at the day's peak interval, then -- like a
    real DR event -- sustains that action across every contiguous interval
    where load is within 40 kW of the day's peak (the "at-risk window"), not
    just the single peak timestamp. This is what actually lowers the day's
    realized peak (and hence the demand charge), rather than shaving one
    instant and letting the very next interval become the new peak.
    """
    peak_idx = int(day_load["total_load_kw"].idxmax())
    peak_row = day_load.iloc[peak_idx]
    peak_val = float(peak_row["total_load_kw"])
    ts = peak_row["timestamp"]
    occ_at_ts = day_occ[day_occ["timestamp"] == ts]

    zone_temps = build_zone_temperatures(peak_row, occ_at_ts)
    occ_known = bool(len(occ_at_ts["occupancy_band"].dropna())) if len(occ_at_ts) else False

    equip_status = {}

    out = run_planner(
        current_load_kw=peak_val,
        predicted_peak_kw=peak_val,
        zone_temperatures=zone_temps,
        equipment_status=equip_status,
        opted_out_zones=set(),
        current_production_load_kw=float(peak_row["production_load_kw"]),
        objective=objective,
        peak_threshold_kw=PEAK_THRESHOLD_KW,
        occupancy_known=occ_known,
    )

    planned_load = day_load.copy()
    reduction = out.plan.total_reduction_kw if out.plan.feasible else 0.0

    # At-risk window: contiguous run of intervals around the peak that are
    # themselves above the peak threshold (i.e. the whole period the demand
    # charge is exposed to), not just the single peak timestamp -- otherwise
    # an untouched neighboring interval simply becomes the new day-peak.
    at_risk_mask = day_load["total_load_kw"] >= PEAK_THRESHOLD_KW
    idxs = at_risk_mask[at_risk_mask].index.to_numpy()
    if len(idxs):
        # find contiguous run containing peak_idx
        run = [peak_idx]
        i = peak_idx - 1
        while i in idxs:
            run.append(i); i -= 1
        i = peak_idx + 1
        while i in idxs:
            run.append(i); i += 1
        window_idx = sorted(run)
    else:
        window_idx = [peak_idx]

    for wi in window_idx:
        applied = min(reduction, max(0.0, planned_load.loc[wi, "total_load_kw"] - 50))
        planned_load.loc[wi, "total_load_kw"] = planned_load.loc[wi, "total_load_kw"] - applied

    hard_violation = 1 if (out.plan.feasible and out.validation.hard_constraints == "FAIL") else 0
    optout_violation = 1 if (out.plan.feasible and out.validation.opt_outs == "FAIL") else 0

    # Comfort-violation minutes attributable to THIS plan: any HVAC action whose
    # projected temperature exceeds the zone's comfort_max, over its assumed
    # duration (30 min default per spec example).
    comfort_violation_minutes = 0
    if out.plan.feasible:
        from planner.facility_config import get_zone
        for a in out.plan.actions:
            if a.equipment_type == "HVAC":
                zone = get_zone(a.zone_id)
                temp = zone_temps.get(a.zone_id)
                if temp is not None:
                    projected = temp + a.reduction_kw * 0.06
                    if projected > zone.comfort_max:
                        comfort_violation_minutes += 30  # default action duration

    return {
        "planned_peak_kw": float(planned_load["total_load_kw"].max()),
        "reduction_kw": reduction,
        "feasible": out.plan.feasible,
        "shortfall_kw": out.plan.shortfall_kw,
        "hard_violation": hard_violation,
        "optout_violation": optout_violation,
        "comfort_violation_minutes": comfort_violation_minutes,
        "planned_load_df": planned_load,
    }


def run_full_evaluation():
    load_df = pd.read_csv(DATA_DIR / "load_timeseries.csv", parse_dates=["timestamp"])
    occ_df = pd.read_csv(DATA_DIR / "occupancy.csv", parse_dates=["timestamp"])

    days = sorted(load_df["timestamp"].dt.date.unique())
    rows = []
    runtime_start = time.time()

    for day in days:
        day_load = load_df[load_df["timestamp"].dt.date == day].reset_index(drop=True)
        day_occ = occ_df[occ_df["timestamp"].dt.date == day]
        scenario = day_load["scenario"].iloc[0]

        baseline = compute_baseline(day_load, day_occ)

        t0 = time.time()
        cost_sim = simulate_day_with_planner(day_load, day_occ, "COST_FIRST")
        cost_runtime = time.time() - t0
        t0 = time.time()
        occ_sim = simulate_day_with_planner(day_load, day_occ, "OCCUPANT_FIRST")
        occ_runtime = time.time() - t0

        cost_dchg = demand_charge(cost_sim["planned_peak_kw"])
        cost_ecost = energy_cost(cost_sim["planned_load_df"])
        occ_dchg = demand_charge(occ_sim["planned_peak_kw"])
        occ_ecost = energy_cost(occ_sim["planned_load_df"])

        rows.append({
            "date": str(day), "scenario": scenario,
            "baseline_peak_kw": baseline.peak_kw,
            "baseline_demand_charge": baseline.demand_charge,
            "baseline_energy_cost": baseline.energy_cost,
            "baseline_comfort_violation_min": baseline.comfort_violation_minutes,
            "cost_first_peak_kw": round(cost_sim["planned_peak_kw"], 1),
            "cost_first_reduction_kw": round(cost_sim["reduction_kw"], 1),
            "cost_first_feasible": cost_sim["feasible"],
            "cost_first_shortfall_kw": cost_sim["shortfall_kw"],
            "cost_first_demand_charge": round(cost_dchg, 2),
            "cost_first_energy_cost": round(cost_ecost, 2),
            "cost_first_comfort_violation_min": cost_sim["comfort_violation_minutes"],
            "cost_first_hard_violation": cost_sim["hard_violation"],
            "cost_first_optout_violation": cost_sim["optout_violation"],
            "cost_first_runtime_s": round(cost_runtime, 4),
            "occupant_first_peak_kw": round(occ_sim["planned_peak_kw"], 1),
            "occupant_first_reduction_kw": round(occ_sim["reduction_kw"], 1),
            "occupant_first_feasible": occ_sim["feasible"],
            "occupant_first_shortfall_kw": occ_sim["shortfall_kw"],
            "occupant_first_demand_charge": round(occ_dchg, 2),
            "occupant_first_energy_cost": round(occ_ecost, 2),
            "occupant_first_comfort_violation_min": occ_sim["comfort_violation_minutes"],
            "occupant_first_hard_violation": occ_sim["hard_violation"],
            "occupant_first_optout_violation": occ_sim["optout_violation"],
            "occupant_first_runtime_s": round(occ_runtime, 4),
        })

    total_runtime = time.time() - runtime_start
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "daily_comparison.csv", index=False)

    # --- forecasting evaluation ---
    hist_pred = historical_average_forecast(load_df)
    hist_errors = forecast_errors(load_df["total_load_kw"].values, hist_pred.values)

    train_df, test_df = train_test_split_time(load_df, test_frac=0.2)
    ml_preds, _model = ml_forecast_train_predict(train_df, test_df)
    ml_errors = forecast_errors(test_df["total_load_kw"].values, ml_preds)

    summary = {
        "n_days": len(days),
        "n_observations": len(load_df),
        "peak_threshold_kw": PEAK_THRESHOLD_KW,
        "n_peak_event_days": int((df["baseline_peak_kw"] > PEAK_THRESHOLD_KW).sum()),
        "peak_reduction_pct_all_days": {
            "cost_first": round(float((1 - df["cost_first_peak_kw"] / df["baseline_peak_kw"]).mean()) * 100, 2),
            "occupant_first": round(float((1 - df["occupant_first_peak_kw"] / df["baseline_peak_kw"]).mean()) * 100, 2),
        },
        "avg_peak_kw": {
            "baseline": round(float(df["baseline_peak_kw"].mean()), 1),
            "cost_first": round(float(df["cost_first_peak_kw"].mean()), 1),
            "occupant_first": round(float(df["occupant_first_peak_kw"].mean()), 1),
        },
        "total_demand_charge": {
            "baseline": round(float(df["baseline_demand_charge"].sum()), 2),
            "cost_first": round(float(df["cost_first_demand_charge"].sum()), 2),
            "occupant_first": round(float(df["occupant_first_demand_charge"].sum()), 2),
        },
        "total_energy_cost": {
            "baseline": round(float(df["baseline_energy_cost"].sum()), 2),
            "cost_first": round(float(df["cost_first_energy_cost"].sum()), 2),
            "occupant_first": round(float(df["occupant_first_energy_cost"].sum()), 2),
        },
        "avg_comfort_violation_min_per_day": {
            "baseline": round(float(df["baseline_comfort_violation_min"].mean()), 2),
            "cost_first": round(float(df["cost_first_comfort_violation_min"].mean()), 2),
            "occupant_first": round(float(df["occupant_first_comfort_violation_min"].mean()), 2),
        },
        "total_hard_violations": {
            "cost_first": int(df["cost_first_hard_violation"].sum()),
            "occupant_first": int(df["occupant_first_hard_violation"].sum()),
        },
        "total_optout_violations": {
            "cost_first": int(df["cost_first_optout_violation"].sum()),
            "occupant_first": int(df["occupant_first_optout_violation"].sum()),
        },
        "infeasible_days": {
            "cost_first": int((~df["cost_first_feasible"]).sum()),
            "occupant_first": int((~df["occupant_first_feasible"]).sum()),
        },
        "avg_planner_runtime_s": {
            "cost_first": round(float(df["cost_first_runtime_s"].mean()), 4),
            "occupant_first": round(float(df["occupant_first_runtime_s"].mean()), 4),
        },
        "forecast_error": {
            "historical_average": hist_errors,
            "random_forest": ml_errors,
        },
        "total_runtime_s": round(total_runtime, 2),
    }

    # Metrics restricted to days where a DR action was actually required
    # (baseline peak above threshold) -- this is the meaningful comparison
    # for "how well does the planner reduce an actual peak event", since
    # most days in this synthetic month never approach the threshold at all.
    event_df = df[df["baseline_peak_kw"] > PEAK_THRESHOLD_KW]
    if len(event_df):
        summary["peak_event_days_only"] = {
            "n_days": len(event_df),
            "peak_reduction_pct": {
                "cost_first": round(float((1 - event_df["cost_first_peak_kw"] / event_df["baseline_peak_kw"]).mean()) * 100, 2),
                "occupant_first": round(float((1 - event_df["occupant_first_peak_kw"] / event_df["baseline_peak_kw"]).mean()) * 100, 2),
            },
            "avg_comfort_violation_min": {
                "cost_first": round(float(event_df["cost_first_comfort_violation_min"].mean()), 2),
                "occupant_first": round(float(event_df["occupant_first_comfort_violation_min"].mean()), 2),
            },
            "avg_reduction_kw": {
                "cost_first": round(float(event_df["cost_first_reduction_kw"].mean()), 1),
                "occupant_first": round(float(event_df["occupant_first_reduction_kw"].mean()), 1),
            },
        }

    with open(RESULTS_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return df, summary

if __name__ == "__main__":
    df, summary = run_full_evaluation()
    print(json.dumps(summary, indent=2))
