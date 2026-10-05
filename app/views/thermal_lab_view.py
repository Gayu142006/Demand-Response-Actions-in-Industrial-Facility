"""
Thermal Modeling & Comfort Lab View.

Interactive lumped-capacitance (1R1C and 2R2C) thermal simulation tool.
Allows operators and engineers to simulate temperature rises under HVAC curtailment,
verify safety margins, and inspect physics parameters.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from planner.thermal_model import (
    simulate_1r1c,
    simulate_2r2c,
    get_thermal_params,
    max_safe_reduction,
    DEFAULT_THERMAL_PARAMS,
)
from planner.facility_config import ZONES, get_zone, ZONE_SAFETY_MARGIN


def render():
    st.title("🌡️ Thermal Modeling & Comfort Lab")
    st.caption("1R1C / 2R2C Lumped-Capacitance Physics Engine — Comfort Guarantee Verification")

    zone_names = [z.zone_id for z in ZONES if z.zone_id in DEFAULT_THERMAL_PARAMS]

    col_params, col_sim = st.columns([1, 2])

    with col_params:
        st.subheader("Simulation Parameters")
        selected_zone = st.selectbox("Select Facility Zone", zone_names, index=0)
        zone_obj = get_zone(selected_zone)
        params = get_thermal_params(selected_zone)

        init_temp = st.slider("Current Zone Temperature (°C)", 18.0, 26.0, float(zone_obj.comfort_min + 2.0), 0.5)
        outdoor_temp = st.slider("Outdoor Ambient Temperature (°C)", 25.0, 45.0, 36.0, 1.0)
        hvac_curtailment = st.slider("HVAC Curtailment (kW electrical)", 0.0, float(params.hvac_capacity_kw / params.hvac_cop), 15.0, 1.0)
        duration_mins = st.select_slider("Event Duration (minutes)", [15, 30, 45, 60, 90, 120], value=45)

        st.divider()
        st.markdown("##### Zone Physical Properties")
        st.write(f"- **Envelope Resistance (R_env):** `{params.R_env:.1f} °C/kW`")
        st.write(f"- **Air Capacitance (C_air):** `{params.C_air:.1f} kWh/°C`")
        st.write(f"- **Wall Capacitance (C_wall):** `{params.C_wall:.1f} kWh/°C`")
        st.write(f"- **Cooling COP:** `{params.hvac_cop:.1f}`")

    with col_sim:
        st.subheader("Temperature Trajectory & Limit Analysis")

        # Run simulations
        p_outdoor = get_thermal_params(selected_zone, T_outdoor=outdoor_temp)
        pred_1r1c = simulate_1r1c(init_temp, p_outdoor, hvac_curtailment, duration_minutes=duration_mins,
                                  comfort_max=zone_obj.comfort_max, safety_limit=zone_obj.comfort_max + ZONE_SAFETY_MARGIN)
        pred_2r2c = simulate_2r2c(init_temp, None, p_outdoor, hvac_curtailment, duration_minutes=duration_mins,
                                  comfort_max=zone_obj.comfort_max, safety_limit=zone_obj.comfort_max + ZONE_SAFETY_MARGIN)

        # Plotly chart
        fig = go.Figure()
        time_x = list(range(len(pred_1r1c.T_trajectory)))
        fig.add_trace(go.Scatter(x=time_x, y=pred_1r1c.T_trajectory, name="1R1C Model (Air node)",
                                 line=dict(color="#3B82F6", width=2)))
        fig.add_trace(go.Scatter(x=time_x, y=pred_2r2c.T_trajectory, name="2R2C Model (Coupled wall mass)",
                                 line=dict(color="#10B981", width=2.5)))

        # Comfort threshold
        fig.add_hline(y=zone_obj.comfort_max, line_dash="dash", line_color="#F59E0B",
                      annotation_text=f"Comfort Ceiling: {zone_obj.comfort_max}°C")
        fig.add_hline(y=zone_obj.comfort_max + ZONE_SAFETY_MARGIN, line_dash="solid", line_color="#EF4444",
                      annotation_text=f"Absolute Safety Limit: {zone_obj.comfort_max + ZONE_SAFETY_MARGIN}°C")

        fig.update_layout(
            title=f"Zone Temperature Dynamics: {selected_zone} ({duration_mins} min curtailment)",
            xaxis_title="Time elapsed (minutes)",
            yaxis_title="Zone Temperature (°C)",
            template="plotly_dark",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Initial Temperature", f"{init_temp:.1f} °C")
        m2.metric("2R2C Peak Temp", f"{pred_2r2c.T_max:.1f} °C",
                  f"{pred_2r2c.T_max - init_temp:+.1f} °C")
        headroom_kw = max_safe_reduction(selected_zone, init_temp, zone_obj.comfort_max, ZONE_SAFETY_MARGIN, duration_mins, outdoor_temp)
        m3.metric("Max Safe Curtailment", f"{headroom_kw:.1f} kW", "Safe capacity")

        if pred_2r2c.T_max <= zone_obj.comfort_max:
            st.success("✅ **Comfort Guaranteed**: Zone stays completely within occupant comfort boundaries throughout event.")
        elif pred_2r2c.T_max <= zone_obj.comfort_max + ZONE_SAFETY_MARGIN:
            st.warning(f"⚠️ **Soft Comfort Excursion**: Exceeds comfort ceiling at {pred_2r2c.time_to_comfort_limit:.0f} min, but strictly within safety margin.")
        else:
            st.error("🚨 **Hard Safety Violation**: Curtailment will breach absolute safety limits! Optimizer hard constraint will block this.")
