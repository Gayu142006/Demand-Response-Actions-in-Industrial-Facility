"""
Plan validator (spec section 26 "Validate plans" / section 57 safety boundary).

Runs a final, independent pass over an already-optimized plan to make
absolutely sure no hard constraint slipped through, and to compute the
PASS/FAIL summary shown to the facility manager (spec section 27).
"""
from dataclasses import dataclass
from planner.constraints import ActionCandidate, FacilityState, validate_hard_constraints


@dataclass
class ValidationSummary:
    hard_constraints: str      # "PASS" | "FAIL"
    opt_outs: str               # "PASS" | "FAIL"
    violations: list


def validate_plan(actions, state: FacilityState) -> ValidationSummary:
    violations = []
    optout_violation = False

    for a in actions:
        cand = ActionCandidate(
            equipment_id=a.equipment_id, zone_id=a.zone_id, equipment_type=a.equipment_type,
            proposed_reduction_kw=a.reduction_kw, is_critical=False, flexibility=a.flexibility,
        )
        result = validate_hard_constraints(cand, state)
        if not result.passed:
            violations.extend([(a.equipment_id, r) for r in result.violated_rules])
        if a.zone_id in state.opted_out_zones:
            optout_violation = True

    return ValidationSummary(
        hard_constraints="PASS" if not violations else "FAIL",
        opt_outs="FAIL" if optout_violation else "PASS",
        violations=violations,
    )
