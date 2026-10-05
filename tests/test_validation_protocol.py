"""
Unit tests for the Stakeholder Validation Protocol and scoring rubric.
"""
import pytest
from validation.protocol import (
    ALL_QUESTIONS,
    FACILITY_MANAGER_QUESTIONS,
    OCCUPANT_ADVOCATE_QUESTIONS,
    Question,
    QuestionResponse,
    ValidationSession,
    normalize_response,
    score_session,
    aggregate_feedback_report,
    generate_design_recommendations,
    run_simulated_validation,
)


def test_questionnaire_coverage():
    assert len(FACILITY_MANAGER_QUESTIONS) >= 10
    assert len(OCCUPANT_ADVOCATE_QUESTIONS) >= 8
    categories = {q.category for q in ALL_QUESTIONS}
    assert "USABILITY" in categories
    assert "TRUST" in categories
    assert "SAFETY" in categories
    assert "PRIVACY" in categories


def test_response_normalization():
    q_likert = Question("T1", "USABILITY", "FACILITY_MANAGER", "Rate usability", "LIKERT_5")
    q_rating = Question("T2", "TRUST", "FACILITY_MANAGER", "Rate trust", "RATING_10")
    q_yesno = Question("T3", "SAFETY", "FACILITY_MANAGER", "Is it safe?", "YES_NO")

    r1 = QuestionResponse("T1", 5, "user1", "FACILITY_MANAGER")
    assert normalize_response(r1, q_likert) == 1.0

    r2 = QuestionResponse("T1", 1, "user1", "FACILITY_MANAGER")
    assert normalize_response(r2, q_likert) == 0.0

    r3 = QuestionResponse("T2", 10, "user1", "FACILITY_MANAGER")
    assert normalize_response(r3, q_rating) == 1.0

    r4 = QuestionResponse("T3", True, "user1", "FACILITY_MANAGER")
    assert normalize_response(r4, q_yesno) == 1.0


def test_session_scoring_and_action_items():
    session = ValidationSession(
        session_id="TEST_001",
        respondent_id="test_fm",
        respondent_role="FACILITY_MANAGER",
        facility_name="Test Plant",
        date="2026-10-05",
    )
    # Low scores on TRUST (1 out of 5)
    session.responses.append(QuestionResponse("FM_T1", False, "test_fm", "FACILITY_MANAGER"))
    session.responses.append(QuestionResponse("FM_T2", 1, "test_fm", "FACILITY_MANAGER"))
    session.responses.append(QuestionResponse("FM_T3", 2, "test_fm", "FACILITY_MANAGER"))

    scored = score_session(session)
    assert scored.overall_score is not None
    assert "TRUST" in scored.category_scores
    assert scored.category_scores["TRUST"] < 50.0
    # Should trigger critical action item for TRUST
    assert any("TRUST" in item for item in scored.action_items)


def test_aggregate_feedback_and_recommendations():
    sim_session = run_simulated_validation()
    report = aggregate_feedback_report([sim_session])
    assert report["n_sessions"] == 1
    assert "overall_score_avg" in report
    assert "category_averages" in report

    recommendations = generate_design_recommendations(report)
    assert len(recommendations) > 0
