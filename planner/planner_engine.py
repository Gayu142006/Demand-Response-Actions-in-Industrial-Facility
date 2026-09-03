"""
Planner engine: ties forecast -> constraints -> optimization -> validation ->
explanation into the single algorithm described in spec section 26.

This module is the one the app and the evaluation experiment both call, so
the demo and the experiment are guaranteed to use identical logic.
"""
from dataclasses import dataclass
from planner.facility_config import PEAK_THRESHOLD_KW
from planner.constraints import FacilityState
from planner.optimizer import optimize, PlanResult
from planner.validator import validate_plan, ValidationSummary


@dataclass
class PlannerOutput:
    current_load_kw: float
    predicted_peak_kw: float
    peak_threshold_kw: float
    required_reduction_kw: float
    plan: PlanResult
    validation: ValidationSummary
    objective: str
    occupancy_confidence: str   # "NORMAL" | "LOW" (missing-data failure case)


def run_planner(
    current_load_kw: float,
    predicted_peak_kw: float,
    zone_temperatures: dict,
    equipment_status: dict,
    opted_out_zones: set,
    current_production_load_kw: float,
    objective: str = "COST_FIRST",
    peak_threshold_kw: float = PEAK_THRESHOLD_KW,
    occupancy_known: bool = True,
) -> PlannerOutput:
    required_reduction = max(0.0, predicted_peak_kw - peak_threshold_kw)

    state = FacilityState(
        zone_temperatures=zone_temperatures,
        equipment_status=equipment_status,
        opted_out_zones=set(opted_out_zones),
        current_production_load_kw=current_production_load_kw,
        occupancy_known=occupancy_known,
    )

    if required_reduction <= 0:
        # No peak-response action needed at all.
        empty_plan = PlanResult(
            feasible=True, actions=[], total_reduction_kw=0.0,
            predicted_peak_kw=predicted_peak_kw, planned_peak_kw=predicted_peak_kw,
            comfort_risk_score=0.0, rejected_candidates=[], shortfall_kw=0.0,
        )
        validation = ValidationSummary(hard_constraints="PASS", opt_outs="PASS", violations=[])
        return PlannerOutput(
            current_load_kw=current_load_kw, predicted_peak_kw=predicted_peak_kw,
            peak_threshold_kw=peak_threshold_kw, required_reduction_kw=0.0,
            plan=empty_plan, validation=validation, objective=objective,
            occupancy_confidence="NORMAL" if occupancy_known else "LOW",
        )

    plan = optimize(
        current_load_kw=current_load_kw,
        predicted_peak_kw=predicted_peak_kw,
        required_reduction_kw=required_reduction,
        state=state,
        objective=objective,
    )
    validation = validate_plan(plan.actions, state) if plan.feasible else \
        ValidationSummary(hard_constraints="PASS", opt_outs="PASS", violations=[])

    return PlannerOutput(
        current_load_kw=current_load_kw, predicted_peak_kw=predicted_peak_kw,
        peak_threshold_kw=peak_threshold_kw, required_reduction_kw=round(required_reduction, 1),
        plan=plan, validation=validation, objective=objective,
        occupancy_confidence="NORMAL" if occupancy_known else "LOW",
    )
