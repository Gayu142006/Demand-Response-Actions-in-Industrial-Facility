import json
from pathlib import Path
import streamlit as st
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "results"


def render():
    st.title("Evaluation")
    st.caption("Baseline vs Cost-First vs Occupant-First, computed by actually running the planner "
               "over the full 30-day synthetic dataset. Run `python experiments/evaluation.py` to refresh.")

    summary_path = RESULTS_DIR / "evaluation_summary.json"
    daily_path = RESULTS_DIR / "daily_comparison.csv"
    if not summary_path.exists():
        st.warning("No evaluation results found yet. Run `python experiments/evaluation.py` first.")
        return

    summary = json.loads(summary_path.read_text())
    daily = pd.read_csv(daily_path)

    st.subheader("Headline numbers")
    c1, c2, c3 = st.columns(3)
    c1.metric("Days simulated", summary["n_days"])
    c2.metric("Days requiring a DR action", summary["n_peak_event_days"])
    if "peak_event_days_only" in summary:
        c3.metric("Peak reduction (event days, cost-first)",
                  f"{summary['peak_event_days_only']['peak_reduction_pct']['cost_first']}%")

    st.subheader("Peak demand (kW)")
    st.bar_chart(pd.DataFrame({
        "Baseline": [summary["avg_peak_kw"]["baseline"]],
        "Cost-first": [summary["avg_peak_kw"]["cost_first"]],
        "Occupant-first": [summary["avg_peak_kw"]["occupant_first"]],
    }, index=["Avg daily peak"]).T)

    st.subheader("Cost")
    cost_df = pd.DataFrame({
        "Demand charge": [summary["total_demand_charge"]["baseline"], summary["total_demand_charge"]["cost_first"],
                           summary["total_demand_charge"]["occupant_first"]],
        "Energy cost": [summary["total_energy_cost"]["baseline"], summary["total_energy_cost"]["cost_first"],
                        summary["total_energy_cost"]["occupant_first"]],
    }, index=["Baseline", "Cost-first", "Occupant-first"])
    st.dataframe(cost_df)

    st.subheader("Comfort & constraint safety")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Avg comfort-violation minutes/day")
        st.json(summary["avg_comfort_violation_min_per_day"])
    with c2:
        st.write("Hard / opt-out violations (target: 0)")
        st.write(f"Hard: {summary['total_hard_violations']} · Opt-out: {summary['total_optout_violations']}")

    if "peak_event_days_only" in summary:
        st.subheader("Peak-event days only (the meaningful comparison)")
        st.caption(f"{summary['peak_event_days_only']['n_days']} of {summary['n_days']} days actually "
                    "exceeded the peak threshold and required a DR action.")
        st.json(summary["peak_event_days_only"])

    st.subheader("Forecast accuracy")
    st.json(summary["forecast_error"])

    st.subheader("Daily detail")
    st.dataframe(daily)
