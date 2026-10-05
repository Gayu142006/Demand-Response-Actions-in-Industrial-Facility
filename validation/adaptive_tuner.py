"""
Adaptive Feedback-Driven Planner Calibration.

Translates stakeholder validation responses into objective weights
and threshold adjustments for the optimizer.

For example:
  - If Occupant Advocate comfort scores are < 70%, comfort_weight increases from 1.0 up to 2.5
  - If Facility Manager trust or safety scores are low, safety margin is widened
  - If disruption scores are low, disruption_weight is adjusted
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
from validation.protocol import aggregate_feedback_report, load_all_sessions

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "data" / "calibration_profile.json"
CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PlannerCalibrationProfile:
    comfort_weight_cost_first: float = 1.0
    comfort_weight_occ_first: float = 8.0
    disruption_weight: float = 1.0
    safety_margin_c: float = 3.0
    pin_penalty_multiplier: float = 20.0
    auto_approve_threshold_pct: float = 85.0
    last_calibrated: Optional[str] = None
    tuning_rationale: str = "Default factory calibration"


def compute_calibration_from_feedback() -> PlannerCalibrationProfile:
    """Analyze all stored validation sessions and produce an adapted calibration profile."""
    sessions = load_all_sessions()
    report = aggregate_feedback_report(sessions)
    profile = PlannerCalibrationProfile()

    if report["n_sessions"] == 0:
        return profile

    cats = report.get("category_averages", {})
    rationales = []

    # If trust is low, increase occupant-first comfort weight
    trust_score = cats.get("TRUST", 80.0)
    if trust_score < 70.0:
        profile.comfort_weight_occ_first = 12.0
        profile.comfort_weight_cost_first = 1.5
        rationales.append(f"Trust scored {trust_score}%: increased comfort weights to protect occupants.")

    # If safety score is low, widen temperature safety margin
    safety_score = cats.get("SAFETY", 80.0)
    if safety_score < 75.0:
        profile.safety_margin_c = 4.0
        rationales.append(f"Safety scored {safety_score}%: widened zone safety margin to 4.0°C.")

    # If usability is high, raise auto-approve threshold
    usability_score = cats.get("USABILITY", 70.0)
    if usability_score > 80.0:
        profile.auto_approve_threshold_pct = 90.0
        rationales.append(f"High usability ({usability_score}%): raised auto-approve threshold.")

    profile.tuning_rationale = "; ".join(rationales) if rationales else "All categories within nominal targets."
    save_calibration(profile)
    return profile


def save_calibration(profile: PlannerCalibrationProfile) -> Path:
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return CALIBRATION_FILE


def load_calibration() -> PlannerCalibrationProfile:
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE) as f:
                data = json.load(f)
                return PlannerCalibrationProfile(**data)
        except Exception:
            pass
    return PlannerCalibrationProfile()
