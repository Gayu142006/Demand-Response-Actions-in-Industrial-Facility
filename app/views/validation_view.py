"""
Stakeholder Validation & User Feedback Portal.

Provides interactive evaluation questionnaires for Facility Managers and
Occupant Advocates, computes scoring rubrics across Usability, Trust, Safety,
Explainability, and Privacy, and feeds back into design recommendations.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
import plotly.graph_objects as go

from validation.protocol import (
    FACILITY_MANAGER_QUESTIONS,
    OCCUPANT_ADVOCATE_QUESTIONS,
    QuestionResponse,
    ValidationSession,
    score_session,
    save_session,
    load_all_sessions,
    aggregate_feedback_report,
    generate_design_recommendations,
)
from validation.adaptive_tuner import compute_calibration_from_feedback, load_calibration


def render():
    st.title("👥 Stakeholder Validation & Feedback Integration")
    st.caption("Live Facility Manager & Occupant Advocate Protocol with Quantitative Rubric")

    tab1, tab2, tab3 = st.tabs(["📝 Conduct Validation Session", "📈 Aggregate Scorecard", "⚙️ Adaptive Calibration"])

    with tab1:
        st.subheader("Interactive Evaluation Form")
        role = st.radio("Respondent Persona", ["FACILITY_MANAGER", "OCCUPANT_ADVOCATE"], horizontal=True)
        user_name = st.text_input("Respondent Name / Anonymized ID", value="Facility-Lead-01")
        facility_name = st.text_input("Facility Name", value="Regional Manufacturing Facility")

        questions = FACILITY_MANAGER_QUESTIONS if role == "FACILITY_MANAGER" else OCCUPANT_ADVOCATE_QUESTIONS

        responses = []
        st.markdown("---")
        for q in questions:
            st.markdown(f"**[{q.category}]** {q.text}")
            if q.scale == "LIKERT_5":
                val = st.slider("Rating (1 = Poor / Disagree, 5 = Excellent / Strongly Agree)", 1, 5, 4, key=f"q_{q.id}")
            elif q.scale == "RATING_10":
                val = st.slider("Rating (1 to 10)", 1, 10, 8, key=f"q_{q.id}")
            elif q.scale == "YES_NO":
                val = st.radio("Response", [True, False], format_func=lambda x: "Yes / Acceptable" if x else "No / Unacceptable", key=f"q_{q.id}")
            else:
                val = st.text_area("Feedback / Comments", key=f"q_{q.id}")

            responses.append(QuestionResponse(
                question_id=q.id,
                response_value=val,
                respondent_id=user_name,
                respondent_role=role,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            st.write("")

        if st.button("Submit & Score Session", type="primary"):
            session = ValidationSession(
                session_id=f"SES_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                respondent_id=user_name,
                respondent_role=role,
                facility_name=facility_name,
                date=datetime.now(timezone.utc).isoformat(),
                responses=responses,
            )
            saved_path = save_session(session)
            st.success(f"Session scored and saved! Overall Score: {session.overall_score}%")
            st.json(session.category_scores)
            if session.action_items:
                st.warning("Action items triggered:")
                for a in session.action_items:
                    st.write(f"- {a}")

    with tab2:
        st.subheader("Multi-Stakeholder Feedback Aggregate")
        all_sessions = load_all_sessions()
        report = aggregate_feedback_report(all_sessions)

        m1, m2, m3 = st.columns(3)
        m1.metric("Completed Sessions", report.get("n_sessions", 0))
        m2.metric("Overall Score Avg", f"{report.get('overall_score_avg', 0)}%")
        m3.metric("Action Items Flagged", len(report.get("unique_action_items", [])))

        cat_avgs = report.get("category_averages", {})
        if cat_avgs:
            categories = list(cat_avgs.keys())
            scores = list(cat_avgs.values())

            fig = go.Figure(data=go.Scatterpolar(
                r=scores + [scores[0]],
                theta=categories + [categories[0]],
                fill='toself',
                marker=dict(color='#3B82F6'),
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                template="plotly_dark",
                height=350,
                title="Cross-Category Performance Radar"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Design Recommendations & Action Items")
        recs = generate_design_recommendations(report)
        for r in recs:
            st.info(f"💡 {r}")

    with tab3:
        st.subheader("Feedback-Driven Optimizer Tuning")
        st.write("Automatically adjust optimizer objective weights and safety margins based on validation scores.")
        curr_cal = load_calibration()

        c1, c2, c3 = st.columns(3)
        c1.metric("Comfort Weight (Occupant-First)", f"{curr_cal.comfort_weight_occ_first:.1f}x")
        c2.metric("Safety Margin", f"{curr_cal.safety_margin_c:.1f} °C")
        c3.metric("Auto-Approve Threshold", f"{curr_cal.auto_approve_threshold_pct:.0f}%")

        st.caption(f"**Rationale:** {curr_cal.tuning_rationale}")

        if st.button("Re-Calibrate Planner from Live Feedback"):
            new_cal = compute_calibration_from_feedback()
            st.success("Planner successfully re-calibrated from latest validation feedback sessions!")
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
