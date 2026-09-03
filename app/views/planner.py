import streamlit as st
from app.state import load_dataset, current_snapshot, live_facility_state
from planner.facility_config import PEAK_THRESHOLD_KW
from planner.planner_engine import run_planner


def render():
    st.title("Demand Response Planner")

    load_df, occ_df = load_dataset()
    day_options = sorted(load_df["timestamp"].dt.date.unique())
    sel_day = st.selectbox("Planning horizon (day)", day_options, index=len(day_options) - 1)
    day_load = load_df[load_df["timestamp"].dt.date == sel_day].reset_index(drop=True)

    threshold = st.number_input("Peak threshold (kW)", value=float(PEAK_THRESHOLD_KW), step=10.0)

    snap = current_snapshot(day_load, occ_df)
    live = live_facility_state()
    predicted_peak = float(day_load["total_load_kw"].max())

    st.info(f"Current demand: **{snap['current_load_kw']:.0f} kW** · "
            f"Predicted peak today: **{predicted_peak:.0f} kW** · Threshold: **{threshold:.0f} kW**")

    planner_state_key = f"{sel_day}_{threshold}_{snap['current_load_kw']}_{len(live['opted_out_zones'])}"
    plans_needed = ("cost_first_plan" not in st.session_state or 
                    st.session_state.get("_last_planner_key") != planner_state_key)

    col_btn, _ = st.columns([1, 4])
    if col_btn.button("🔄 Recalculate Plans", type="primary") or plans_needed:
        cost_first = run_planner(
            current_load_kw=snap["current_load_kw"], predicted_peak_kw=predicted_peak,
            zone_temperatures=snap["zone_temperatures"], equipment_status=live["equipment_status"],
            opted_out_zones=live["opted_out_zones"], current_production_load_kw=snap["production_load_kw"],
            objective="COST_FIRST", peak_threshold_kw=threshold, occupancy_known=snap["occupancy_known"],
        )
        occupant_first = run_planner(
            current_load_kw=snap["current_load_kw"], predicted_peak_kw=predicted_peak,
            zone_temperatures=snap["zone_temperatures"], equipment_status=live["equipment_status"],
            opted_out_zones=live["opted_out_zones"], current_production_load_kw=snap["production_load_kw"],
            objective="OCCUPANT_FIRST", peak_threshold_kw=threshold, occupancy_known=snap["occupancy_known"],
        )
        st.session_state["cost_first_plan"] = cost_first
        st.session_state["occupant_first_plan"] = occupant_first
        st.session_state["_last_planner_key"] = planner_state_key
        st.session_state.pop("plan_explanation", None)

    cost_first = st.session_state["cost_first_plan"]
    occupant_first = st.session_state["occupant_first_plan"]

    if snap["occupancy_known"] is False:
        st.warning("Occupancy data is UNKNOWN for this interval — planner confidence is LOW and "
                   "HVAC reductions are conservatively reduced (Failure Case 1).")

    col1, col2 = st.columns(2)
    for col, out, label in [(col1, cost_first, "COST-FIRST"), (col2, occupant_first, "OCCUPANT-FIRST")]:
        with col:
            st.subheader(label)
            if out.required_reduction_kw <= 0:
                st.success("No action needed — within threshold.")
                continue
            if not out.plan.feasible:
                st.error(f"NO FEASIBLE PLAN\n\nRequired reduction: {out.required_reduction_kw:.0f} kW\n\n"
                          f"{out.plan.reason_if_infeasible}")
                st.write("Suggested actions: reduce peak target · extend response window · "
                          "request authorized override · schedule additional flexible load later.")
                continue
            for a in out.plan.actions:
                with st.expander(f"{a.equipment_id} — {a.reduction_kw:.0f} kW"):
                    for r in a.reasons:
                        st.write(f"✓ {r}")
            st.metric("Predicted peak → planned peak",
                      f"{out.plan.predicted_peak_kw:.0f} → {out.plan.planned_peak_kw:.0f} kW")
            reduction_pct = (out.plan.predicted_peak_kw - out.plan.planned_peak_kw) / out.plan.predicted_peak_kw * 100
            st.write(f"Reduction: {out.plan.total_reduction_kw:.0f} kW ({reduction_pct:.1f}%)")
            st.write(f"Hard constraints: **{out.validation.hard_constraints}** · "
                      f"Opt-outs: **{out.validation.opt_outs}**")
            if out.plan.shortfall_kw > 0.5:
                st.warning(f"Shortfall remaining: {out.plan.shortfall_kw:.0f} kW")
            if out.plan.rejected_candidates:
                with st.expander("Rejected candidates (hard-constraint failures)"):
                    for eid, rules in out.plan.rejected_candidates:
                        st.write(f"- {eid}: {', '.join(rules)}")

    st.divider()
    st.subheader("Trade-off summary")
    def summarize(out):
        if out.required_reduction_kw <= 0 or not out.plan.feasible:
            return {"Peak (kW)": out.plan.planned_peak_kw, "Reduction %": 0.0,
                    "Comfort risk score": out.plan.comfort_risk_score}
        pct = (out.plan.predicted_peak_kw - out.plan.planned_peak_kw) / out.plan.predicted_peak_kw * 100
        return {"Peak (kW)": out.plan.planned_peak_kw, "Reduction %": round(pct, 1),
                "Comfort risk score": out.plan.comfort_risk_score}
    import pandas as pd
    st.dataframe(pd.DataFrame({"Cost-first": summarize(cost_first), "Occupant-first": summarize(occupant_first)}))
    st.caption("Higher comfort-risk score = more occupant discomfort risk. The facility manager "
               "chooses the plan that fits the current situation (spec section 25).")

    # --- AI DR Copilot Section ---
    st.divider()
    st.subheader("🤖 AI Demand-Response Copilot & Decision Explainer")

    from planner.llm_assistant import generate_plan_explanation, chat_with_copilot, LLMConfig
    llm_cfg = st.session_state.get("llm_config", LLMConfig())

    provider_name = llm_cfg.provider.replace("_", " ").title()
    st.caption(f"Powered by: **{provider_name}**" + (f" ({llm_cfg.model_name})" if llm_cfg.model_name else ""))

    tab_explain, tab_chat = st.tabs(["📋 Executive Plan Analysis", "💬 Interactive Copilot Chat"])

    with tab_explain:
        if "plan_explanation" not in st.session_state or st.button("🔄 Refresh AI Explanation"):
            with st.spinner("Analyzing demand response scenario with AI Copilot..."):
                explanation = generate_plan_explanation(cost_first, occupant_first, llm_cfg)
                st.session_state["plan_explanation"] = explanation

        st.markdown(st.session_state["plan_explanation"])

    with tab_chat:
        st.write("Ask your AI Copilot about curtailment decisions, comfort impacts, or safety constraints.")

        if "copilot_messages" not in st.session_state:
            st.session_state["copilot_messages"] = [
                {"role": "assistant", "content": "Hello! I am your AI Demand Response Copilot. How can I help you evaluate or refine today's demand response plan?"}
            ]

        # Suggested prompt chips
        quick_cols = st.columns(3)
        if quick_cols[0].button("⚖️ Compare Strategies"):
            st.session_state["copilot_messages"].append({"role": "user", "content": "What is the difference between the Cost-First and Occupant-First plans for this scenario?"})
            ans = chat_with_copilot(st.session_state["copilot_messages"], cost_first, occupant_first, llm_cfg)
            st.session_state["copilot_messages"].append({"role": "assistant", "content": ans})
            st.rerun()

        if quick_cols[1].button("🛡️ Check Safety Constraints"):
            st.session_state["copilot_messages"].append({"role": "user", "content": "How are hard safety constraints and opt-outs protected in these plans?"})
            ans = chat_with_copilot(st.session_state["copilot_messages"], cost_first, occupant_first, llm_cfg)
            st.session_state["copilot_messages"].append({"role": "assistant", "content": ans})
            st.rerun()

        if quick_cols[2].button("🌡️ HVAC vs Battery Logic"):
            st.session_state["copilot_messages"].append({"role": "user", "content": "Why was HVAC or battery chosen for curtailment instead of other equipment?"})
            ans = chat_with_copilot(st.session_state["copilot_messages"], cost_first, occupant_first, llm_cfg)
            st.session_state["copilot_messages"].append({"role": "assistant", "content": ans})
            st.rerun()

        for msg in st.session_state["copilot_messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask a question about this plan (e.g. Why was Zone 1 chosen?)...")
        if user_input:
            st.session_state["copilot_messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("AI Copilot is thinking..."):
                    response = chat_with_copilot(st.session_state["copilot_messages"], cost_first, occupant_first, llm_cfg)
                    st.write(response)
            st.session_state["copilot_messages"].append({"role": "assistant", "content": response})

