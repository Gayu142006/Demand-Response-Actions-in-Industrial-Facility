import streamlit as st
import pandas as pd
from app.state import load_dataset, current_snapshot, live_facility_state
from planner.facility_config import PEAK_THRESHOLD_KW
from planner.planner_engine import run_planner
from app.database import get_active_opt_outs


def render():
    st.title("Facility Dashboard")

    load_df, occ_df = load_dataset()
    day_options = sorted(load_df["timestamp"].dt.date.unique())
    sel_day = st.selectbox("Day", day_options, index=len(day_options) - 1)
    day_load = load_df[load_df["timestamp"].dt.date == sel_day].reset_index(drop=True)
    day_occ = occ_df[occ_df["timestamp"].dt.date == sel_day]

    snap = current_snapshot(day_load, occ_df)
    live = live_facility_state()

    predicted_peak = float(day_load["total_load_kw"].max())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current demand", f"{snap['current_load_kw']:.0f} kW")
    col2.metric("Predicted peak (today)", f"{predicted_peak:.0f} kW")
    col3.metric("Peak threshold", f"{PEAK_THRESHOLD_KW:.0f} kW")
    risk = "HIGH" if predicted_peak > PEAK_THRESHOLD_KW else "LOW"
    col4.metric("Peak risk", risk)

    st.caption(f"Scenario of the day: **{snap['scenario']}** · snapshot at {snap['timestamp']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Occupancy", snap["occupancy_band"])
    comfort_ok = all(
        v is None or (18 <= v <= 30) for v in snap["zone_temperatures"].values()
    )
    c2.metric("Comfort", "GOOD" if comfort_ok else "AT RISK")
    active_optouts = get_active_opt_outs()
    c3.metric("Active opt-outs", len(active_optouts))
    if active_optouts:
        st.caption("Opted-out zones: " + ", ".join(active_optouts))

    st.divider()
    st.subheader("Load profile")
    st.line_chart(day_load.set_index("timestamp")["total_load_kw"])

    st.divider()
    st.subheader("Recommended actions (cost-first, quick preview)")
    out = run_planner(
        current_load_kw=snap["current_load_kw"],
        predicted_peak_kw=predicted_peak,
        zone_temperatures=snap["zone_temperatures"],
        equipment_status=live["equipment_status"],
        opted_out_zones=live["opted_out_zones"],
        current_production_load_kw=snap["production_load_kw"],
        objective="COST_FIRST",
        occupancy_known=snap["occupancy_known"],
    )
    if out.required_reduction_kw <= 0:
        st.success("No demand-response action needed — predicted peak is within threshold.")
    elif not out.plan.feasible:
        st.error(f"NO FEASIBLE PLAN — required reduction {out.required_reduction_kw:.0f} kW, "
                  f"shortfall {out.plan.shortfall_kw:.0f} kW. {out.plan.reason_if_infeasible}")
    else:
        for a in out.plan.actions:
            st.write(f"**{a.equipment_id}** — {a.reduction_kw:.0f} kW ({a.flexibility.lower()} flexibility)")
        st.info(f"Expected peak: {out.predicted_peak_kw:.0f} → {out.plan.planned_peak_kw:.0f} kW "
                f"(reduction {out.plan.total_reduction_kw:.0f} kW)")
        if out.plan.shortfall_kw > 0.5:
            st.warning(f"Shortfall of {out.plan.shortfall_kw:.0f} kW remains after constraints. "
                       "See Planner page for full detail and the occupant-first alternative.")
    st.caption("Go to the **Planner** page to generate and compare cost-first vs occupant-first plans in full.")
