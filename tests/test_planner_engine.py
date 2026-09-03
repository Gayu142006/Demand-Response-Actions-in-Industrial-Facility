from planner.planner_engine import run_planner


STD_TEMPS = {"OFFICE_A": 23, "WAREHOUSE": 24, "PRODUCTION": 23,
             "CONTROL_ROOM": 21, "UTILITY": 25, "BATTERY_FLEX": 22}


def test_no_action_needed_when_below_threshold():
    out = run_planner(
        current_load_kw=600, predicted_peak_kw=680, zone_temperatures=STD_TEMPS,
        equipment_status={}, opted_out_zones=set(), current_production_load_kw=260,
        objective="COST_FIRST", peak_threshold_kw=750,
    )
    assert out.required_reduction_kw == 0
    assert out.plan.feasible
    assert out.plan.actions == []


def test_worked_example_matches_spec_expectations():
    """Spec section: current 735kW, predicted peak 824kW, threshold 750kW ->
    required reduction 74kW, plan should close (or nearly close) the gap."""
    out = run_planner(
        current_load_kw=735, predicted_peak_kw=824, zone_temperatures=STD_TEMPS,
        equipment_status={}, opted_out_zones=set(), current_production_load_kw=260,
        objective="COST_FIRST", peak_threshold_kw=750,
    )
    assert out.required_reduction_kw == 74.0
    assert out.plan.feasible
    assert out.plan.planned_peak_kw <= 751
    assert out.validation.hard_constraints == "PASS"


# --- Named failure cases (spec section "failure cases") --------------------

def test_failure_case_missing_occupancy_data():
    """Occupancy sensor data missing -> planner marks LOW confidence and is
    conservative with HVAC (does not silently assume normal occupancy)."""
    out = run_planner(
        current_load_kw=735, predicted_peak_kw=824,
        zone_temperatures={**STD_TEMPS, "OFFICE_A": None},
        equipment_status={}, opted_out_zones=set(), current_production_load_kw=260,
        objective="COST_FIRST", peak_threshold_kw=750, occupancy_known=False,
    )
    assert out.occupancy_confidence == "LOW"
    office_actions = [a for a in out.plan.actions if a.zone_id == "OFFICE_A" and a.equipment_type == "HVAC"]
    assert all(a.reduction_kw <= 10 for a in office_actions)


def test_failure_case_equipment_failure_excluded_with_reason():
    out = run_planner(
        current_load_kw=735, predicted_peak_kw=824, zone_temperatures=STD_TEMPS,
        equipment_status={"COMPRESSOR_B": "MAINTENANCE"}, opted_out_zones=set(),
        current_production_load_kw=260, objective="COST_FIRST", peak_threshold_kw=750,
    )
    ids = {a.equipment_id for a in out.plan.actions}
    assert "COMPRESSOR_B" not in ids
    rejected_ids = {r[0] for r in out.plan.rejected_candidates}
    assert "COMPRESSOR_B" in rejected_ids


def test_failure_case_multiple_simultaneous_optouts():
    out = run_planner(
        current_load_kw=735, predicted_peak_kw=824, zone_temperatures=STD_TEMPS,
        equipment_status={}, opted_out_zones={"WAREHOUSE", "OFFICE_A"},
        current_production_load_kw=260, objective="COST_FIRST", peak_threshold_kw=750,
    )
    zones_used = {a.zone_id for a in out.plan.actions}
    assert "WAREHOUSE" not in zones_used
    assert "OFFICE_A" not in zones_used
    # Still must find SOME plan from remaining zones
    assert out.plan.feasible


def test_failure_case_extreme_heat_conflicting_signals():
    hot_temps = {"OFFICE_A": 37, "WAREHOUSE": 39, "PRODUCTION": 38,
                  "CONTROL_ROOM": 22, "UTILITY": 40, "BATTERY_FLEX": 38}
    out = run_planner(
        current_load_kw=735, predicted_peak_kw=824, zone_temperatures=hot_temps,
        equipment_status={}, opted_out_zones=set(), current_production_load_kw=260,
        objective="COST_FIRST", peak_threshold_kw=750,
    )
    # Must not push any zone further into unsafe territory via HVAC reduction
    assert out.validation.hard_constraints == "PASS"


def test_failure_case_no_feasible_plan_reports_shortfall_not_silent_failure():
    out = run_planner(
        current_load_kw=1200, predicted_peak_kw=1300, zone_temperatures=STD_TEMPS,
        equipment_status={"COMPRESSOR_A": "MAINTENANCE", "COMPRESSOR_B": "MAINTENANCE"},
        opted_out_zones={"WAREHOUSE", "OFFICE_A", "UTILITY", "BATTERY_FLEX"},
        current_production_load_kw=260, objective="COST_FIRST", peak_threshold_kw=750,
    )
    assert out.plan.feasible is False
    assert out.plan.shortfall_kw > 0
    assert out.plan.reason_if_infeasible != ""


def test_objective_never_overrides_hard_constraints():
    """Both objectives must produce plans that pass hard-constraint validation --
    occupant-first cannot 'relax' a safety constraint to protect comfort."""
    for objective in ("COST_FIRST", "OCCUPANT_FIRST"):
        out = run_planner(
            current_load_kw=735, predicted_peak_kw=824, zone_temperatures=STD_TEMPS,
            equipment_status={"COMPRESSOR_B": "MAINTENANCE"}, opted_out_zones={"WAREHOUSE"},
            current_production_load_kw=260, objective=objective, peak_threshold_kw=750,
        )
        assert out.validation.hard_constraints == "PASS"
        assert out.validation.opt_outs == "PASS"
