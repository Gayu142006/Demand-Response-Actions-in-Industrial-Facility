"""
Stakeholder Validation Protocol (spec sections on multi-source validation).

Provides structured interview scripts, scoring rubrics, and a programmatic
feedback-integration pipeline so that real user feedback can be captured,
scored, and fed back into the planner's configuration (e.g., adjusting
comfort weights, changing approval-flow verbosity, or flagging missing
equipment-specific explanations).

Status: PROTOCOL READY — awaiting real user sessions.
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

FEEDBACK_DIR = Path(__file__).resolve().parent / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Questionnaire definitions
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    category: str              # USABILITY, TRUST, SAFETY, EXPLAINABILITY, PRIVACY
    audience: str              # FACILITY_MANAGER | OCCUPANT_ADVOCATE | BOTH
    text: str
    scale: str                 # LIKERT_5 | YES_NO | FREE_TEXT | RATING_10
    weight: float = 1.0        # importance weight for aggregate scoring


FACILITY_MANAGER_QUESTIONS: List[Question] = [
    # --- Usability ---
    Question("FM_U1", "USABILITY", "FACILITY_MANAGER",
             "Walk through one full day's Dashboard → Planner → Approvals flow for a "
             "real peak event. Does the recommended action list match what you would "
             "actually do?", "LIKERT_5", weight=2.0),
    Question("FM_U2", "USABILITY", "FACILITY_MANAGER",
             "How quickly could you approve or reject a plan during a live peak event? "
             "(1=too slow, 5=fast enough)", "LIKERT_5", weight=1.5),
    Question("FM_U3", "USABILITY", "FACILITY_MANAGER",
             "Is the side-by-side Cost-First vs Occupant-First comparison useful for your "
             "decision process, or would you prefer a single recommended plan?", "LIKERT_5"),

    # --- Trust ---
    Question("FM_T1", "TRUST", "FACILITY_MANAGER",
             "Would you trust this system's recommendation enough to approve it in under "
             "30 seconds during a real peak event?", "YES_NO", weight=2.0),
    Question("FM_T2", "TRUST", "FACILITY_MANAGER",
             "Do the per-action reasons (e.g., 'HIGH flexibility', 'no opt-out conflict') "
             "give you enough confidence, or do you need more equipment-specific detail?",
             "LIKERT_5", weight=1.5),
    Question("FM_T3", "TRUST", "FACILITY_MANAGER",
             "Rate your overall trust in the system's safety guarantees (1=no trust, "
             "10=complete trust).", "RATING_10", weight=2.0),

    # --- Safety ---
    Question("FM_S1", "SAFETY", "FACILITY_MANAGER",
             "Show a NO FEASIBLE PLAN case. Is the explanation and the suggested "
             "next-steps list actually useful, or does it need to be more specific to "
             "your equipment?", "LIKERT_5", weight=2.0),
    Question("FM_S2", "SAFETY", "FACILITY_MANAGER",
             "Review the override audit log. Is 'authorized_user + role + reason + "
             "affected_constraint + duration' enough detail for your compliance process?",
             "YES_NO", weight=1.5),
    Question("FM_S3", "SAFETY", "FACILITY_MANAGER",
             "Would you ever want the system to override a hard safety constraint "
             "automatically? If so, under what conditions?", "FREE_TEXT"),

    # --- Explainability ---
    Question("FM_E1", "EXPLAINABILITY", "FACILITY_MANAGER",
             "Does the AI Copilot's Executive Plan Analysis clearly explain WHY specific "
             "equipment was chosen for curtailment?", "LIKERT_5", weight=1.5),
    Question("FM_E2", "EXPLAINABILITY", "FACILITY_MANAGER",
             "Are the rejected-candidate reasons (e.g., 'critical_equipment_must_remain_on') "
             "clear enough, or do they need plain-language translations?", "LIKERT_5"),

    # --- Completeness ---
    Question("FM_C1", "USABILITY", "FACILITY_MANAGER",
             "What equipment or zone types are MISSING from the current model that your "
             "real facility has?", "FREE_TEXT"),
    Question("FM_C2", "USABILITY", "FACILITY_MANAGER",
             "What additional metrics would you need to see on the Dashboard before "
             "approving a plan?", "FREE_TEXT"),
]


OCCUPANT_ADVOCATE_QUESTIONS: List[Question] = [
    # --- Usability ---
    Question("OA_U1", "USABILITY", "OCCUPANT_ADVOCATE",
             "Show the opt-out flow. Is TEMPORARY_OPT_OUT easy to set before a known "
             "sensitive period (e.g., a client visit, a sensitive production run)?",
             "LIKERT_5", weight=1.5),
    Question("OA_U2", "USABILITY", "OCCUPANT_ADVOCATE",
             "Is the 3-tier preference model (PARTICIPATE / ONLY_IF_NECESSARY / OPT_OUT) "
             "clear and meaningful to you?", "LIKERT_5"),

    # --- Privacy ---
    Question("OA_P1", "PRIVACY", "OCCUPANT_ADVOCATE",
             "Does the system ever feel like it's tracking you individually, or does the "
             "zone-level/band-level framing feel appropriately anonymous?", "YES_NO", weight=2.0),
    Question("OA_P2", "PRIVACY", "OCCUPANT_ADVOCATE",
             "Review docs/privacy.md. What is missing from the privacy commitments?",
             "FREE_TEXT", weight=1.5),

    # --- Trust ---
    Question("OA_T1", "TRUST", "OCCUPANT_ADVOCATE",
             "Do you trust that your TEMPORARY_OPT_OUT will actually be respected during "
             "a DR event?", "LIKERT_5", weight=2.0),
    Question("OA_T2", "TRUST", "OCCUPANT_ADVOCATE",
             "Rate your comfort with the idea of automated HVAC curtailment in your "
             "workspace (1=very uncomfortable, 10=fully comfortable).", "RATING_10"),

    # --- Comfort ---
    Question("OA_C1", "USABILITY", "OCCUPANT_ADVOCATE",
             "Is 30 minutes an acceptable maximum duration for HVAC curtailment during a "
             "peak event?", "YES_NO"),
    Question("OA_C2", "USABILITY", "OCCUPANT_ADVOCATE",
             "Would you want to be notified BEFORE an HVAC curtailment happens, or is "
             "after-the-fact notification sufficient?", "FREE_TEXT"),
]


ALL_QUESTIONS = FACILITY_MANAGER_QUESTIONS + OCCUPANT_ADVOCATE_QUESTIONS


# ---------------------------------------------------------------------------
# Response capture and scoring
# ---------------------------------------------------------------------------

@dataclass
class QuestionResponse:
    question_id: str
    response_value: Any          # int for LIKERT_5/RATING_10, bool for YES_NO, str for FREE_TEXT
    respondent_id: str           # anonymized or role-based
    respondent_role: str         # FACILITY_MANAGER | OCCUPANT_ADVOCATE
    timestamp: str = ""
    notes: str = ""


@dataclass
class ValidationSession:
    session_id: str
    respondent_id: str
    respondent_role: str
    facility_name: str
    date: str
    responses: List[QuestionResponse] = field(default_factory=list)
    overall_score: Optional[float] = None
    category_scores: Dict[str, float] = field(default_factory=dict)
    action_items: List[str] = field(default_factory=list)


def normalize_response(response: QuestionResponse, question: Question) -> float:
    """Normalize any response type to a 0..1 scale for aggregation."""
    if question.scale == "LIKERT_5":
        return max(0.0, min(1.0, (float(response.response_value) - 1) / 4))
    elif question.scale == "RATING_10":
        return max(0.0, min(1.0, (float(response.response_value) - 1) / 9))
    elif question.scale == "YES_NO":
        return 1.0 if response.response_value else 0.0
    elif question.scale == "FREE_TEXT":
        # Free text responses contribute qualitatively, not to the numeric score
        return None
    return 0.5


def score_session(session: ValidationSession) -> ValidationSession:
    """Compute aggregate and per-category scores for a completed session."""
    question_map = {q.id: q for q in ALL_QUESTIONS}

    category_totals: Dict[str, float] = {}
    category_weights: Dict[str, float] = {}

    for resp in session.responses:
        q = question_map.get(resp.question_id)
        if q is None:
            continue
        norm = normalize_response(resp, q)
        if norm is None:
            continue
        category_totals[q.category] = category_totals.get(q.category, 0.0) + norm * q.weight
        category_weights[q.category] = category_weights.get(q.category, 0.0) + q.weight

    session.category_scores = {
        cat: round(category_totals[cat] / category_weights[cat] * 100, 1)
        for cat in category_totals if category_weights[cat] > 0
    }

    if category_weights:
        total_w = sum(category_weights.values())
        total_s = sum(category_totals.values())
        session.overall_score = round(total_s / total_w * 100, 1)
    else:
        session.overall_score = 0.0

    # Generate action items from low-scoring categories
    session.action_items = []
    for cat, score in session.category_scores.items():
        if score < 60:
            session.action_items.append(
                f"CRITICAL: {cat} scored {score}% — requires immediate design revision"
            )
        elif score < 75:
            session.action_items.append(
                f"MODERATE: {cat} scored {score}% — schedule improvement for next iteration"
            )

    return session


def save_session(session: ValidationSession) -> Path:
    """Persist a scored validation session to disk."""
    session = score_session(session)
    path = FEEDBACK_DIR / f"session_{session.session_id}.json"
    with open(path, "w") as f:
        json.dump(asdict(session), f, indent=2, default=str)
    return path


def load_all_sessions() -> List[ValidationSession]:
    """Load all persisted validation sessions."""
    sessions = []
    for p in sorted(FEEDBACK_DIR.glob("session_*.json")):
        with open(p) as f:
            data = json.load(f)
            sessions.append(ValidationSession(**data))
    return sessions


def aggregate_feedback_report(sessions: List[ValidationSession]) -> Dict[str, Any]:
    """Aggregate across multiple sessions to produce a summary report."""
    if not sessions:
        return {"n_sessions": 0, "message": "No validation sessions recorded yet."}

    all_category_scores: Dict[str, List[float]] = {}
    all_overall: List[float] = []
    all_action_items: List[str] = []

    for s in sessions:
        if s.overall_score is not None:
            all_overall.append(s.overall_score)
        for cat, score in (s.category_scores or {}).items():
            all_category_scores.setdefault(cat, []).append(score)
        all_action_items.extend(s.action_items or [])

    return {
        "n_sessions": len(sessions),
        "overall_score_avg": round(sum(all_overall) / len(all_overall), 1) if all_overall else None,
        "category_averages": {
            cat: round(sum(scores) / len(scores), 1)
            for cat, scores in all_category_scores.items()
        },
        "unique_action_items": list(set(all_action_items)),
        "respondent_roles": {
            "FACILITY_MANAGER": sum(1 for s in sessions if s.respondent_role == "FACILITY_MANAGER"),
            "OCCUPANT_ADVOCATE": sum(1 for s in sessions if s.respondent_role == "OCCUPANT_ADVOCATE"),
        },
    }


# ---------------------------------------------------------------------------
# Feedback → design integration
# ---------------------------------------------------------------------------

def generate_design_recommendations(report: Dict[str, Any]) -> List[str]:
    """Translate aggregate validation feedback into concrete design changes."""
    recommendations = []
    cats = report.get("category_averages", {})

    if cats.get("TRUST", 100) < 70:
        recommendations.append(
            "TRUST is low — add detailed per-action provenance (which sensor reading, "
            "which constraint rule) to the Planner UI. Consider adding a 'simulation mode' "
            "where managers can see what would have happened under different scenarios."
        )
    if cats.get("EXPLAINABILITY", 100) < 70:
        recommendations.append(
            "EXPLAINABILITY is low — replace constraint rule IDs with plain-language "
            "descriptions (e.g., 'critical_equipment_must_remain_on' → 'This equipment "
            "cannot be curtailed because it is classified as life-safety critical')."
        )
    if cats.get("USABILITY", 100) < 70:
        recommendations.append(
            "USABILITY is low — simplify the approval workflow. Consider auto-approving "
            "plans that match historical patterns, with one-click override."
        )
    if cats.get("PRIVACY", 100) < 80:
        recommendations.append(
            "PRIVACY concerns flagged — review data retention policies and consider "
            "adding a consent management dashboard."
        )
    if cats.get("SAFETY", 100) < 80:
        recommendations.append(
            "SAFETY confidence is below target — add more prominent visual indicators "
            "for hard-constraint protection status in the UI."
        )
    if not recommendations:
        recommendations.append(
            "All categories scored above thresholds. Focus on iterative polish."
        )

    return recommendations


# ---------------------------------------------------------------------------
# Simulated validation session (for demonstration/testing)
# ---------------------------------------------------------------------------

def run_simulated_validation() -> ValidationSession:
    """
    Create a simulated validation session with realistic but clearly synthetic
    responses. This is for demonstrating the protocol flow, NOT for claiming
    real stakeholder validation.
    """
    now = datetime.now(timezone.utc).isoformat()
    session = ValidationSession(
        session_id="SIM_001",
        respondent_id="simulated_facility_manager",
        respondent_role="FACILITY_MANAGER",
        facility_name="Synthetic Industrial Facility A",
        date=now,
    )

    simulated_responses = {
        "FM_U1": 4, "FM_U2": 4, "FM_U3": 3,
        "FM_T1": True, "FM_T2": 4, "FM_T3": 7,
        "FM_S1": 3, "FM_S2": True, "FM_S3": "Only for non-safety constraints like lighting during off-hours",
        "FM_E1": 4, "FM_E2": 3,
        "FM_C1": "Missing: variable-speed drives on pumps, solar PV inverter curtailment",
        "FM_C2": "Power factor, reactive power, transformer loading percentage",
    }

    for q in FACILITY_MANAGER_QUESTIONS:
        session.responses.append(QuestionResponse(
            question_id=q.id,
            response_value=simulated_responses.get(q.id, "N/A"),
            respondent_id=session.respondent_id,
            respondent_role=session.respondent_role,
            timestamp=now,
            notes="SIMULATED — not from a real user",
        ))

    return score_session(session)


if __name__ == "__main__":
    session = run_simulated_validation()
    path = save_session(session)
    print(f"Saved simulated session to {path}")
    print(f"Overall score: {session.overall_score}%")
    print(f"Category scores: {json.dumps(session.category_scores, indent=2)}")
    if session.action_items:
        print("Action items:")
        for item in session.action_items:
            print(f"  - {item}")
    recommendations = generate_design_recommendations(
        aggregate_feedback_report([session])
    )
    print("\nDesign recommendations:")
    for r in recommendations:
        print(f"  → {r}")
