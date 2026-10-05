"""
Equipment Fatigue & Lifecycle Wear View.

Monitors cycling counts, cumulative curtailed energy, and wear indices to
prevent premature failure of compressors, pumps, and battery systems during DR.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from planner.equipment_fatigue import EquipmentFatigueTracker
from planner.facility_config import EQUIPMENT

tracker = EquipmentFatigueTracker()


def render():
    st.title("⚙️ Equipment Fatigue & Lifecycle Wear")
    st.caption("Thermal & Mechanical Stress Tracking — Rotational Curtailment Fairness")

    # Initialize tracker with known equipment if empty
    for eq in EQUIPMENT:
        tracker.get_or_create(eq.equipment_id, eq.equipment_type)

    fleet = tracker.get_fleet_status()
    df = pd.DataFrame(fleet)

    col1, col2, col3 = st.columns(3)
    avg_fatigue = df["fatigue_index"].mean() if len(df) else 0.0
    total_cycles = df["cycles_this_month"].sum() if len(df) else 0
    total_kwh = df["total_curtailed_kwh"].sum() if len(df) else 0.0

    col1.metric("Fleet Average Fatigue", f"{avg_fatigue:.1f} / 100")
    col2.metric("Total DR Activations This Month", f"{total_cycles}")
    col3.metric("Cumulative Curtailed Energy", f"{total_kwh:.1f} kWh")

    st.divider()

    st.subheader("Equipment Fleet Status & Wear Multipliers")

    fig = px.bar(
        df,
        x="equipment_id",
        y="fatigue_index",
        color="fatigue_index",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 100],
        title="Fatigue Index by Equipment (Higher = Greater Wear)",
        labels={"fatigue_index": "Fatigue Index (0-100)", "equipment_id": "Equipment"},
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        df[[
            "equipment_id",
            "equipment_type",
            "cycles_this_month",
            "max_recommended_cycles_per_month",
            "total_curtailed_kwh",
            "fatigue_index",
            "wear_penalty_multiplier",
        ]],
        use_container_width=True,
    )

    st.markdown("#### Rotational Curtailment Balancing")
    st.info(
        "Equipment with a fatigue index over 50.0 receives an automatically scaling wear penalty in "
        "the optimizer objective. This automatically shifts curtailment burdens to alternate redundant equipment "
        "(e.g. alternating between Compressor A and Compressor B) to extend machine lifecycle."
    )
