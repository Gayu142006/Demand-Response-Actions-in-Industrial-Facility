"""
Optimization engine.

Uses OR-Tools CP-SAT (real solver, actually installed and imported -- not a
stub) to select how much reduction to take from each available piece of
equipment. Hard constraints are pre-filtered before the solver ever sees a
candidate (an equipment item that fails a hard constraint is simply removed
from the decision variables, never "penalized into submission").

Two objectives per spec sections 23/24:
  - COST_FIRST:      lambda1 (peak) >> lambda3 (comfort)
  - OCCUPANT_FIRST:  lambda3 (comfort) >> lambda1 (peak)
"""
from dataclasses import dataclass, field
from typing import Optional
from ortools.sat.python import cp_model

from planner.facility_config import EQUIPMENT, get_zone
from planner.constraints import (
    ActionCandidate, FacilityState, validate_hard_constraints,
    comfort_penalty, disruption_penalty, preference_penalty,
)

# Objective weights (lambda1..lambda4), spec section 23/24.
WEIGHTS = {
    "COST_FIRST": dict(peak=10.0, energy=1.0, comfort=1.0, disruption=1.0),
    "OCCUPANT_FIRST": dict(peak=2.0, energy=1.0, comfort=8.0, disruption=3.0),
}

SCALE = 100  # cp-sat needs integers; scale kW to hundredths for resolution


@dataclass
class PlannedAction:
    equipment_id: str
    zone_id: str
    equipment_type: str
    reduction_kw: float
    flexibility: str
    reasons: list


@dataclass
class PlanResult:
    feasible: bool
    actions: list
    total_reduction_kw: float
    predicted_peak_kw: float
    planned_peak_kw: float
    comfort_risk_score: float
    rejected_candidates: list       # [(equipment_id, violated_rules)]
    shortfall_kw: float = 0.0
    reason_if_infeasible: str = ""


def build_candidates(state: FacilityState):
    """Build one ActionCandidate per piece of equipment, respecting max_reduction_kw."""
    candidates = []
    for e in EQUIPMENT:
        if e.max_reduction_kw <= 0:
            continue
        candidates.append(ActionCandidate(
            equipment_id=e.equipment_id,
            zone_id=e.zone_id,
            equipment_type=e.equipment_type,
            proposed_reduction_kw=e.max_reduction_kw,  # upper bound; solver picks 0..max
            is_critical=e.is_critical,
            flexibility=e.flexibility,
        ))
    return candidates


def optimize(
    current_load_kw: float,
    predicted_peak_kw: float,
    required_reduction_kw: float,
    state: FacilityState,
    objective: str = "COST_FIRST",
) -> PlanResult:
    assert objective in WEIGHTS
    weights = WEIGHTS[objective]

    candidates = build_candidates(state)

    feasible_equipment = []   # list of (Equipment, max_reduction_kw, comfort_pen, disrupt_pen, pref_pen)
    rejected = []

    for e in EQUIPMENT:
        if e.max_reduction_kw <= 0:
            continue
        # Evaluate hard constraints at the equipment's maximum possible reduction --
        # if even the smallest useful reduction fails, it is excluded entirely.
        test_candidate = ActionCandidate(
            equipment_id=e.equipment_id, zone_id=e.zone_id, equipment_type=e.equipment_type,
            proposed_reduction_kw=e.min_reduction_kw or 1.0, is_critical=e.is_critical,
            flexibility=e.flexibility,
        )
        result = validate_hard_constraints(test_candidate, state)
        if not result.passed:
            rejected.append((e.equipment_id, result.violated_rules))
            continue

        # Determine the actual safe maximum for this equipment (may be capped
        # below its nominal max by e.g. temperature headroom under extreme heat).
        safe_max = e.max_reduction_kw
        if e.equipment_type == "HVAC":
            zone = get_zone(e.zone_id)
            temp = state.zone_temperatures.get(e.zone_id)
            if temp is not None:
                # headroom before hitting absolute safety limit, translated back to kW
                from planner.facility_config import ZONE_SAFETY_MARGIN
                headroom_c = (zone.comfort_max + ZONE_SAFETY_MARGIN) - temp
                headroom_kw = max(0.0, headroom_c / 0.06)
                safe_max = min(safe_max, headroom_kw)
            else:
                safe_max = min(safe_max, 10.0)  # missing-data conservative cap

        if safe_max < e.min_reduction_kw:
            rejected.append((e.equipment_id, ["insufficient_safe_headroom"]))
            continue

        cand = ActionCandidate(
            equipment_id=e.equipment_id, zone_id=e.zone_id, equipment_type=e.equipment_type,
            proposed_reduction_kw=safe_max, is_critical=e.is_critical, flexibility=e.flexibility,
        )
        c_pen = comfort_penalty(cand, state)
        d_pen = disruption_penalty(cand)
        p_pen = preference_penalty(cand, occupant_first=(objective == "OCCUPANT_FIRST"))
        feasible_equipment.append((e, safe_max, c_pen, d_pen, p_pen))

    if not feasible_equipment:
        return PlanResult(
            feasible=False, actions=[], total_reduction_kw=0.0,
            predicted_peak_kw=predicted_peak_kw, planned_peak_kw=predicted_peak_kw,
            comfort_risk_score=0.0, rejected_candidates=rejected,
            shortfall_kw=required_reduction_kw,
            reason_if_infeasible="No equipment passed hard-constraint validation.",
        )

    model = cp_model.CpModel()
    reduction_vars = []
    for e, safe_max, c_pen, d_pen, p_pen in feasible_equipment:
        lo = 0
        hi = int(round(safe_max * SCALE))
        v = model.NewIntVar(lo, hi, f"reduction_{e.equipment_id}")
        reduction_vars.append(v)

    total_reduction = sum(reduction_vars)

    # Soft target: try to hit required_reduction_kw, but never force it if
    # infeasible under hard constraints alone (shortfall is allowed and reported).
    required_scaled = int(round(required_reduction_kw * SCALE))
    shortfall = model.NewIntVar(0, required_scaled if required_scaled > 0 else 0, "shortfall")
    model.Add(total_reduction + shortfall >= required_scaled)

    # Objective: minimize (peak shortfall, weighted) + energy proxy + comfort + disruption
    penalty_terms = []
    for (e, safe_max, c_pen, d_pen, p_pen), v in zip(feasible_equipment, reduction_vars):
        # penalty per unit of reduction taken from this equipment (scaled)
        unit_penalty = (weights["comfort"] * c_pen + weights["disruption"] * d_pen + p_pen)
        unit_penalty_int = max(0, int(round(unit_penalty)))
        penalty_terms.append(unit_penalty_int * v)

    model.Minimize(
        weights["peak"] * SCALE * shortfall
        + sum(penalty_terms)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return PlanResult(
            feasible=False, actions=[], total_reduction_kw=0.0,
            predicted_peak_kw=predicted_peak_kw, planned_peak_kw=predicted_peak_kw,
            comfort_risk_score=0.0, rejected_candidates=rejected,
            shortfall_kw=required_reduction_kw,
            reason_if_infeasible="Solver could not find a feasible allocation.",
        )

    actions = []
    total_kw = 0.0
    total_comfort_risk = 0.0
    for (e, safe_max, c_pen, d_pen, p_pen), v in zip(feasible_equipment, reduction_vars):
        kw = solver.Value(v) / SCALE
        if kw >= max(0.5, e.min_reduction_kw * 0.25):  # ignore negligible allocations
            reasons = []
            reasons.append(f"{e.flexibility} flexibility")
            if c_pen < 1.0:
                reasons.append("no meaningful comfort risk")
            else:
                reasons.append(f"comfort risk score {c_pen:.1f}")
            if e.zone_id not in state.opted_out_zones:
                reasons.append("no opt-out conflict")
            reasons.append(f"{kw:.0f} kW available reduction")
            actions.append(PlannedAction(
                equipment_id=e.equipment_id, zone_id=e.zone_id, equipment_type=e.equipment_type,
                reduction_kw=round(kw, 1), flexibility=e.flexibility, reasons=reasons,
            ))
            total_kw += kw
            total_comfort_risk += c_pen * (kw / max(safe_max, 1e-6))

    planned_peak = predicted_peak_kw - total_kw
    shortfall_val = solver.Value(shortfall) / SCALE

    return PlanResult(
        feasible=True,
        actions=sorted(actions, key=lambda a: -a.reduction_kw),
        total_reduction_kw=round(total_kw, 1),
        predicted_peak_kw=predicted_peak_kw,
        planned_peak_kw=round(planned_peak, 1),
        comfort_risk_score=round(total_comfort_risk, 1),
        rejected_candidates=rejected,
        shortfall_kw=round(shortfall_val, 1),
        reason_if_infeasible="" if shortfall_val <= 0.5 else
            f"Target reduction not fully reachable; shortfall of {shortfall_val:.1f} kW remains "
            f"after respecting comfort and operational constraints.",
    )
