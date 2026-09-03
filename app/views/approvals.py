import json
import streamlit as st
from app.database import save_approved_plan, save_override, get_overrides


def render():
    st.title("Approvals & Overrides")

    if "cost_first_plan" not in st.session_state:
        from app.state import load_dataset, current_snapshot, live_facility_state
        from planner.facility_config import PEAK_THRESHOLD_KW
        from planner.planner_engine import run_planner

        load_df, occ_df = load_dataset()
        day_options = sorted(load_df["timestamp"].dt.date.unique())
        sel_day = day_options[-1]
        day_load = load_df[load_df["timestamp"].dt.date == sel_day].reset_index(drop=True)
        snap = current_snapshot(day_load, occ_df)
        live = live_facility_state()
        predicted_peak = float(day_load["total_load_kw"].max())

        st.session_state["cost_first_plan"] = run_planner(
            current_load_kw=snap["current_load_kw"], predicted_peak_kw=predicted_peak,
            zone_temperatures=snap["zone_temperatures"], equipment_status=live["equipment_status"],
            opted_out_zones=live["opted_out_zones"], current_production_load_kw=snap["production_load_kw"],
            objective="COST_FIRST", peak_threshold_kw=float(PEAK_THRESHOLD_KW), occupancy_known=snap["occupancy_known"],
        )
        st.session_state["occupant_first_plan"] = run_planner(
            current_load_kw=snap["current_load_kw"], predicted_peak_kw=predicted_peak,
            zone_temperatures=snap["zone_temperatures"], equipment_status=live["equipment_status"],
            opted_out_zones=live["opted_out_zones"], current_production_load_kw=snap["production_load_kw"],
            objective="OCCUPANT_FIRST", peak_threshold_kw=float(PEAK_THRESHOLD_KW), occupancy_known=snap["occupancy_known"],
        )

    choice = st.radio("Select plan to review", ["Cost-first", "Occupant-first"], horizontal=True)
    out = st.session_state["cost_first_plan"] if choice == "Cost-first" else st.session_state["occupant_first_plan"]

    if not out.plan.feasible:
        st.error("This plan is infeasible — nothing to approve. Consider an authorized override below, "
                  "or return to the Planner page to relax the peak target.")
    else:
        st.write(f"**{choice}** plan — planned peak {out.plan.planned_peak_kw:.0f} kW, "
                  f"reduction {out.plan.total_reduction_kw:.0f} kW")
        for a in out.plan.actions:
            st.write(f"- {a.equipment_id}: {a.reduction_kw:.0f} kW")

        approver = st.text_input("Approver name / role", value="facility_manager")
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ APPROVE"):
            save_approved_plan(out.objective, out.plan.predicted_peak_kw, out.plan.planned_peak_kw,
                                out.plan.total_reduction_kw, "APPROVED", approver,
                                json.dumps([a.__dict__ for a in out.plan.actions]))
            st.success("Plan approved and logged.")
        if c2.button("✏️ MODIFY"):
            st.info("Modification recorded as a manual override below (adjust the plan's actions on "
                      "the Planner page, then re-approve).")
        if c3.button("❌ REJECT"):
            save_approved_plan(out.objective, out.plan.predicted_peak_kw, out.plan.planned_peak_kw,
                                out.plan.total_reduction_kw, "REJECTED", approver, "[]")
            st.warning("Plan rejected and logged.")

    st.divider()
    st.subheader("Authorized override")
    st.caption("Overrides can relax a soft/preference constraint under authorization — hard "
               "safety constraints are never overridden by this workflow (design principle 5.2).")
    with st.form("override_form"):
        user = st.text_input("Authorized user", value="facility_manager")
        role = st.selectbox("Role", ["Facility Manager", "Admin"])
        reason = st.text_input("Override reason", value="Emergency grid event")
        constraint = st.selectbox("Constraint affected", ["Preferred comfort", "Preferred equipment schedule",
                                                            "Noise preference", "Operational preference"])
        duration = st.number_input("Duration (minutes)", value=30, step=15)
        action = st.text_input("Action authorized", value="Extend HVAC reduction on Office A")
        submit = st.form_submit_button("Log override")
    if submit:
        save_override(user, role, reason, constraint, duration, action)
        st.success("Override logged to audit trail.")

    st.divider()
    st.subheader("Audit log")
    for o in get_overrides():
        st.write(f"`{o['timestamp'][:19]}` — **{o['authorized_user']}** ({o['role']}) — "
                  f"{o['action']} — reason: {o['reason']} — affects: {o['affected_constraint']} "
                  f"— duration: {o['duration_min']} min")
