"""
Synthetic data generator for the Occupant-Aware Demand Response Planner.

This is a ground-truth "world model", not a hand-picked CSV. It simulates:
  - a diurnal load curve per zone (production, HVAC, lighting, compressors,
    pumps, warehouse, battery) at 15-minute resolution
  - aggregated zone-level occupancy bands
  - outdoor/zone temperature driving comfort and HVAC load
  - equipment availability (maintenance windows)
  - opt-out events
  - one of 10 named scenario types per day, deterministic given a seed

The same generated dataset is used for the "dataset generation" deliverable
and for the baseline-vs-planner evaluation experiment, so both are backed by
one consistent model (per spec section 41/61/62).

No real production data or live telemetry exists for this project, so all
numbers below are synthetic and clearly generated (never hand-picked to hit
target metrics).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from planner.facility_config import ZONES, EQUIPMENT, energy_tariff_for_hour

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS = 30
INTERVALS_PER_DAY = 96  # 15-minute resolution
START_DATE = pd.Timestamp("2026-06-01")

# One named scenario per day, cycling deterministically (spec section 62).
SCENARIO_CYCLE = [
    "NORMAL", "NORMAL", "HOT", "HIGH_OCCUPANCY", "NORMAL",
    "LOW_OCCUPANCY", "NORMAL", "HIGH_PRODUCTION", "NORMAL", "HOT",
    "EQUIPMENT_FAILURE", "NORMAL", "MULTIPLE_OPTOUTS", "NORMAL", "EXTREME_HEAT",
    "NORMAL", "MISSING_DATA", "NORMAL", "HIGH_OCCUPANCY", "OFFLINE",
    "NORMAL", "HOT", "NORMAL", "LOW_OCCUPANCY", "NORMAL",
    "HIGH_PRODUCTION", "EQUIPMENT_FAILURE", "NORMAL", "EXTREME_HEAT", "NORMAL",
]


def diurnal_shape(hour: float) -> float:
    """Base diurnal occupancy/load shape in [0,1], per spec section 61 pattern."""
    if hour < 6:
        return 0.15
    if hour < 9:
        return 0.15 + (hour - 6) / 3 * 0.55       # ramp up
    if hour < 12:
        return 0.70 + (hour - 9) / 3 * 0.20        # high load
    if hour < 14:
        return 0.85
    if hour < 17:
        return 0.90 + (hour - 14) / 3 * 0.08        # afternoon peak
    if hour < 21:
        return 0.95 - (hour - 17) / 4 * 0.55        # decline
    return 0.20


def occupancy_band(value: float) -> str:
    if value < 0.35:
        return "LOW"
    if value < 0.70:
        return "MEDIUM"
    return "HIGH"


def generate():
    rng = np.random.default_rng(42)  # reproducible

    rows_load = []
    rows_occ = []
    rows_equipment = []
    rows_optout = []
    day_scenarios = []

    equip_status = {e.equipment_id: "AVAILABLE" for e in EQUIPMENT}
    equip_maintenance_until = {}

    for day_idx in range(N_DAYS):
        date = START_DATE + pd.Timedelta(days=day_idx)
        scenario = SCENARIO_CYCLE[day_idx % len(SCENARIO_CYCLE)]
        day_scenarios.append({"date": date.date().isoformat(), "scenario": scenario})

        # Scenario-level modifiers.
        # `hvac_strain` represents how far normal HVAC operation is pushed
        # away from each zone's comfort mid-point (0 = perfectly held at
        # mid-range under normal HVAC, larger = HVAC struggling to keep up
        # with outdoor heat load -- this is what drives comfort violations
        # in the no-intervention baseline).
        occ_mult = 1.0
        hvac_strain = 0.0
        production_mult = 1.0
        force_missing_data_hours = set()
        force_offline = False
        optout_zones_today = []

        if scenario == "HOT":
            hvac_strain = 2.5
        elif scenario == "EXTREME_HEAT":
            hvac_strain = 9.0
        elif scenario == "HIGH_OCCUPANCY":
            occ_mult = 1.25
        elif scenario == "LOW_OCCUPANCY":
            occ_mult = 0.55
        elif scenario == "HIGH_PRODUCTION":
            production_mult = 1.45
        elif scenario == "EQUIPMENT_FAILURE":
            equip_maintenance_until[("COMPRESSOR_B", day_idx)] = True
        elif scenario == "MULTIPLE_OPTOUTS":
            optout_zones_today = ["OFFICE_A", "WAREHOUSE", "UTILITY"]
        elif scenario == "MISSING_DATA":
            force_missing_data_hours = set(rng.choice(range(24), size=3, replace=False))
        elif scenario == "OFFLINE":
            force_offline = True

        for interval in range(INTERVALS_PER_DAY):
            ts = date + pd.Timedelta(minutes=15 * interval)
            hour = ts.hour + ts.minute / 60

            shape = diurnal_shape(hour)
            noise = rng.normal(0, 0.03)
            occ_value = float(np.clip(shape * occ_mult + noise, 0.03, 1.0))

            # Zone temperature: held near the comfort mid-point under normal
            # HVAC operation (this IS the baseline -- HVAC still runs
            # normally when no demand-response action is taken), with
            # occupancy-driven internal gain and scenario-driven HVAC strain
            # pushing the zone toward (or past) its comfort ceiling.
            zone_temp = {}
            for z in ZONES:
                midpoint = (z.comfort_min + z.comfort_max) / 2
                occ_gain = (1.2 if z.zone_type in ("OFFICE", "CONTROL_ROOM", "PRODUCTION") else 0.3) * occ_value
                zone_temp[z.zone_id] = round(midpoint + occ_gain + hvac_strain + rng.normal(0, 0.5), 2)

            # equipment status for this interval
            for e in EQUIPMENT:
                if equip_maintenance_until.get((e.equipment_id, day_idx)) and 10 <= hour < 15:
                    status = "MAINTENANCE"
                else:
                    status = "AVAILABLE"
                rows_equipment.append({
                    "timestamp": ts, "equipment_id": e.equipment_id, "zone_id": e.zone_id,
                    "status": status,
                })

            # opt-outs
            for zid in optout_zones_today:
                if 13 <= hour < 16:
                    rows_optout.append({
                        "timestamp": ts, "zone_id": zid,
                        "preference": "TEMPORARY_OPT_OUT", "reason": "Scheduled activity",
                    })

            # load per zone
            production_load = 300 * shape * production_mult + rng.normal(0, 4)
            hvac_load = 0
            lighting_load = 0
            for z in ZONES:
                zt = zone_temp[z.zone_id]
                overshoot = max(0, zt - z.comfort_max)
                zone_hvac = 7 + overshoot * 6 + 11 * occ_value
                hvac_load += zone_hvac
            lighting_load = 18 * (0.4 + 0.6 * occ_value)
            compressor_load = 155 * shape * production_mult + rng.normal(0, 5)
            warehouse_load = 78 * (0.5 + 0.5 * occ_value)
            pump_load = 32 * (0.5 + 0.4 * shape)
            battery_load = 48 * (1.0 if 22 <= hour or hour < 6 else 0.3)  # charges off-peak

            total_load = (production_load + hvac_load + lighting_load +
                          compressor_load + warehouse_load + pump_load + battery_load)
            total_load = max(150, total_load)

            occ_band_val = occ_value if hour not in force_missing_data_hours else None

            rows_load.append({
                "timestamp": ts,
                "total_load_kw": round(total_load, 1),
                "production_load_kw": round(max(production_load, 0), 1),
                "hvac_load_kw": round(max(hvac_load, 0), 1),
                "lighting_load_kw": round(max(lighting_load, 0), 1),
                "compressor_load_kw": round(max(compressor_load, 0), 1),
                "warehouse_load_kw": round(max(warehouse_load, 0), 1),
                "pump_load_kw": round(max(pump_load, 0), 1),
                "battery_load_kw": round(max(battery_load, 0), 1),
                "occupancy": None if occ_band_val is None else round(occ_band_val, 3),
                "scenario": scenario,
                "offline": force_offline,
                "tariff_per_kwh": energy_tariff_for_hour(int(hour)),
                **{f"temp_{zid}": t for zid, t in zone_temp.items()},
            })

            for z in ZONES:
                band_val = occ_value if hour not in force_missing_data_hours else None
                rows_occ.append({
                    "timestamp": ts, "zone_id": z.zone_id,
                    "occupancy_band": None if band_val is None else occupancy_band(band_val),
                    "temperature_c": zone_temp[z.zone_id],
                })

    df_load = pd.DataFrame(rows_load)
    df_occ = pd.DataFrame(rows_occ)
    df_equip = pd.DataFrame(rows_equipment)
    df_optout = pd.DataFrame(rows_optout)
    df_scenarios = pd.DataFrame(day_scenarios)

    df_load.to_csv(OUT_DIR / "load_timeseries.csv", index=False)
    df_occ.to_csv(OUT_DIR / "occupancy.csv", index=False)
    df_equip.to_csv(OUT_DIR / "equipment_status.csv", index=False)
    df_optout.to_csv(OUT_DIR / "opt_outs.csv", index=False)
    df_scenarios.to_csv(OUT_DIR / "day_scenarios.csv", index=False)

    print(f"Generated {len(df_load)} load intervals across {N_DAYS} days")
    print(f"Generated {len(df_occ)} zone-occupancy rows")
    print(f"Generated {len(df_equip)} equipment-status rows")
    print(f"Generated {len(df_optout)} opt-out rows")
    print(f"Wrote files to {OUT_DIR}")
    return df_load, df_occ, df_equip, df_optout, df_scenarios


if __name__ == "__main__":
    generate()
