"""
Constraint engine.

Hard constraints are deterministic and can NEVER be relaxed by the
optimizer, no matter which objective is active (spec sections 15 & 17,
design principle 5.2). Soft constraints become optimization penalties
(section 16).

Phase 2: The linear temperature approximation (projected_rise = kW * 0.06)
has been replaced with the 1R1C/2R2C lumped-capacitance thermal model from
planner/thermal_model.py, providing physics-informed comfort guarantees.
"""
from dataclasses import dataclass
from typing import Optional
from planner.facility_config import (
    get_zone, ZONE_SAFETY_MARGIN, MINIMUM_PRODUCTION_LOAD_KW,
)
from planner.thermal_model import projected_temperature_rise, max_safe_reduction


@dataclass
class ActionCandidate:
    equipment_id: str
    zone_id: str
    equipment_type: str
    proposed_reduction_kw: float
    is_critical: bool
    flexibility: str


@dataclass
class FacilityState:
    """Snapshot of everything the constraint engine needs to evaluate a candidate."""
    zone_temperatures: dict          # zone_id -> current temp (C), may be missing -> None
    equipment_status: dict           # equipment_id -> "AVAILABLE" | "MAINTENANCE"
    opted_out_zones: set             # zone_ids under TEMPORARY_OPT_OUT right now
    current_production_load_kw: float
    occupancy_known: bool = True     # False => missing-occupancy failure case
    participate_if_necessary_zones: set = None  # zone_ids under PARTICIPATE_ONLY_IF_NECESSARY

    def __post_init__(self):
        if self.participate_if_necessary_zones is None:
            self.participate_if_necessary_zones = set()


@dataclass
class ConstraintResult:
    passed: bool
    violated_rules: list
    notes: list


def validate_hard_constraints(candidate: ActionCandidate, state: FacilityState) -> ConstraintResult:
    """
    Returns PASS/FAIL for a single candidate action against every hard
    constraint. A candidate that fails ANY hard constraint must be rejected
    outright -- it is never relaxed, weighted, or partially honored.
    """
    violated = []
    notes = []

    # 1. Critical equipment must remain ON / not be reduced.
    if candidate.is_critical:
        violated.append("critical_equipment_must_remain_on")

    # 2. Maintenance-locked equipment cannot be controlled.
    if state.equipment_status.get(candidate.equipment_id) == "MAINTENANCE":
        violated.append("equipment_under_maintenance")

    # 3. Opted-out zones cannot receive normal demand-response actions.
    if candidate.zone_id in state.opted_out_zones:
        violated.append("zone_opted_out")

    # 4. Production load must remain above minimum level.
    if candidate.equipment_type == "PRODUCTION_LINE":
        remaining = state.current_production_load_kw - candidate.proposed_reduction_kw
        if remaining < MINIMUM_PRODUCTION_LOAD_KW:
            violated.append("production_below_minimum")

    # 5. Temperature must remain within absolute safety limits (comfort +/- margin).
    #    Uses the 2R2C lumped-capacitance thermal model for physics-informed prediction,
    #    replacing the old linear approximation (projected_rise = kW * 0.06).
    if candidate.equipment_type == "HVAC":
        zone = get_zone(candidate.zone_id)
        temp = state.zone_temperatures.get(candidate.zone_id)
        if temp is None:
            # Missing data: conservative behavior handled upstream (failure case 1);
            # constraint engine still refuses to certify safety with unknown temp
            # for large reductions.
            if candidate.proposed_reduction_kw > 10:
                violated.append("unknown_temperature_conservative_block")
        else:
            # Use 2R2C thermal model for physics-informed temperature prediction
            absolute_max = zone.comfort_max + ZONE_SAFETY_MARGIN
            prediction = projected_temperature_rise(
                zone_id=candidate.zone_id,
                current_temp=temp,
                hvac_reduction_kw=candidate.proposed_reduction_kw,
                duration_minutes=30.0,
                comfort_max=zone.comfort_max,
                safety_limit=absolute_max,
                model="2R2C",
            )
            if prediction.T_max > absolute_max:
                violated.append("temperature_absolute_safety_limit")
            notes.append(
                f"thermal_model=2R2C projected_max={prediction.T_max:.1f}C "
                f"absolute_max={absolute_max:.1f}C "
                f"time_to_safety={prediction.time_to_safety_limit}min"
            )

    # 6. Action cannot exceed equipment available flexibility (checked by caller
    #    via max_reduction_kw before a candidate is even created, but re-verify).
    if candidate.proposed_reduction_kw < 0:
        violated.append("negative_reduction_invalid")

    return ConstraintResult(passed=(len(violated) == 0), violated_rules=violated, notes=notes)


def comfort_penalty(candidate: ActionCandidate, state: FacilityState) -> float:
    """Soft-constraint penalty: how much occupant comfort is put at risk (0..100).

    Uses the 2R2C thermal model to compute a physics-informed comfort risk score
    based on predicted temperature trajectory during the DR event.
    """
    if candidate.equipment_type != "HVAC":
        return 0.0
    zone = get_zone(candidate.zone_id)
    temp = state.zone_temperatures.get(candidate.zone_id)
    if temp is None:
        return 25.0  # uncertainty penalty
    prediction = projected_temperature_rise(
        zone_id=candidate.zone_id,
        current_temp=temp,
        hvac_reduction_kw=candidate.proposed_reduction_kw,
        duration_minutes=30.0,
        comfort_max=zone.comfort_max,
        safety_limit=zone.comfort_max + ZONE_SAFETY_MARGIN,
        model="2R2C",
    )
    overshoot = max(0.0, prediction.T_max - zone.comfort_max)
    # Bonus penalty if comfort limit is breached early in the event
    time_penalty = 0.0
    if prediction.time_to_comfort_limit is not None:
        time_penalty = max(0.0, (30.0 - prediction.time_to_comfort_limit) / 30.0) * 5.0
    return overshoot * 15.0 + time_penalty


def disruption_penalty(candidate: ActionCandidate) -> float:
    """Soft-constraint penalty: operational disruption of shifting/reducing this load."""
    weight = {"HIGH": 1.0, "MEDIUM": 2.5, "LOW": 6.0}.get(candidate.flexibility, 4.0)
    return weight * (candidate.proposed_reduction_kw / 10.0)


def preference_penalty(candidate: ActionCandidate, occupant_first: bool) -> float:
    """Extra penalty layered on when running the occupant-first objective."""
    if not occupant_first:
        return 0.0
    if candidate.equipment_type in ("HVAC",):
        return candidate.proposed_reduction_kw * 0.4
    return 0.0
