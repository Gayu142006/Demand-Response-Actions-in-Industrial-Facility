# Failure Modes

Each case below is implemented in `planner/`, exercised in
`tests/test_planner_engine.py` / `tests/test_optimizer.py`, and represented
as a scenario day in the synthetic dataset (`scripts/generate_data.py`).

## 1. Missing occupancy data
**Behavior:** `run_planner(..., occupancy_known=False)` sets
`occupancy_confidence = "LOW"` on the output. HVAC candidates in zones with
an unknown current temperature are hard-blocked above a 10kW reduction
(`constraints.py`, rule `unknown_temperature_conservative_block`) rather
than assuming a "safe" default temperature.
**Test:** `test_failure_case_missing_occupancy_data`.
**Known limitation:** the conservative 10kW cap is a fixed heuristic, not
learned from historical safe-margins; a production system would want this
calibrated per zone from real sensor-outage history.

## 2. Equipment failure / maintenance lockout
**Behavior:** any equipment whose `equipment_status == "MAINTENANCE"` is
excluded from the candidate set entirely (rule `equipment_under_maintenance`)
and listed in `PlanResult.rejected_candidates` with the reason, so the
manager can see *why* an obvious lever wasn't used.
**Test:** `test_failure_case_equipment_failure_excluded_with_reason`.

## 3. Multiple simultaneous opt-outs
**Behavior:** every opted-out zone's equipment is excluded (rule
`zone_opted_out`). The optimizer still searches the remaining equipment for
a feasible plan; it does not treat multiple opt-outs as an automatic
"no plan" case.
**Test:** `test_failure_case_multiple_simultaneous_optouts`.
**Known limitation:** the current model does not distinguish "would
strongly prefer not to" from "must not" opt-outs (see spec's
`PARTICIPATE_ONLY_IF_NECESSARY` tier) as a genuinely graduated soft
constraint -- today it's binary (opted out zones are always excluded from
non-emergency actions). This is a real simplification for the prototype's
scope, not an oversight; extending it is one line item in
`docs/scope_and_limitations.md`.

## 4. Extreme heat / conflicting signals (cost pressure vs. comfort safety)
**Behavior:** for each HVAC candidate, the safe reduction ceiling is
recomputed from actual temperature headroom to the absolute safety limit
(`comfort_max + 3°C`). Under extreme heat this ceiling collapses toward
zero, so the optimizer is structurally pushed toward non-HVAC equipment
(compressors, pumps, batteries, lighting) automatically -- this is not a
special-cased "if extreme_heat" branch, it falls out of the constraint math.
**Test:** `test_failure_case_extreme_heat_conflicting_signals`,
`test_extreme_heat_avoids_hvac_and_shifts_to_other_equipment`.

## 5. No feasible plan exists
**Behavior:** when every candidate fails hard constraints, or the solver's
best allocation still can't reach the target, `PlanResult.feasible` is
`False` or `shortfall_kw > 0` respectively, with a specific
`reason_if_infeasible` string. The app surfaces this as an explicit error
with next-step suggestions (relax target / extend window / request
override / schedule additional flexible load) -- it never silently returns
an empty "plan" that looks like success.
**Test:** `test_no_feasible_plan_when_everything_blocked`,
`test_failure_case_no_feasible_plan_reports_shortfall_not_silent_failure`.

## 6. Offline field update conflicting with server state
**Behavior:** `app/database.save_field_observation()` compares a locally
queued equipment-status observation against `equipment_server_state`; a
disagreement is marked `CONFLICT` and is *not* auto-applied even after
`sync_queued_observations()` runs -- it stays flagged until a manager
explicitly resolves it via `resolve_conflict()`.
**Test:** `test_conflict_detected_when_equipment_status_disagrees`,
`test_sync_queued_does_not_clear_conflicts`.

## 7. Objective cannot override a hard constraint
**Behavior:** both `COST_FIRST` and `OCCUPANT_FIRST` objectives only ever
change the *weights* applied to already-hard-constraint-passing candidates
(`optimizer.WEIGHTS`). Neither objective can cause a rejected candidate to
re-enter the candidate pool.
**Test:** `test_objective_never_overrides_hard_constraints`.

## What the evaluation run actually revealed (not hand-picked)

Running `experiments/evaluation.py` over the 30-day synthetic dataset
surfaced a real, non-obvious finding: on the 4 days that actually crossed
the peak threshold (2 EXTREME_HEAT, 2 HIGH_PRODUCTION), **cost-first and
occupant-first produced identical plans.** This is not a bug -- under those
scenarios, HVAC headroom is too small to be useful (extreme heat) or the
entire available flexible capacity is needed just to approach the target
(high production), so there is no slack left for the objective weighting to
express a preference. The two objectives only diverge on moderate DR events
where multiple equipment combinations could each close the gap (verified
manually in `tests/test_optimizer.py::test_occupant_first_produces_lower_or_equal_comfort_risk_than_cost_first`
and interactively in the Planner page). See
`experiments/results/evaluation_summary.json` for the exact numbers.
