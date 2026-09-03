from planner.constraints import FacilityState
from planner.optimizer import optimize


def standard_state(**overrides):
    base = dict(
        zone_temperatures={"OFFICE_A": 23, "WAREHOUSE": 24, "PRODUCTION": 23,
                            "CONTROL_ROOM": 21, "UTILITY": 25, "BATTERY_FLEX": 22},
        equipment_status={},
        opted_out_zones=set(),
        current_production_load_kw=260,
        occupancy_known=True,
    )
    base.update(overrides)
    return FacilityState(**base)


def test_optimizer_meets_target_when_capacity_available():
    state = standard_state()
    result = optimize(current_load_kw=735, predicted_peak_kw=824, required_reduction_kw=74,
                       state=state, objective="COST_FIRST")
    assert result.feasible
    assert result.shortfall_kw <= 0.5
    assert result.planned_peak_kw <= 750.5


def test_optimizer_never_selects_critical_equipment():
    state = standard_state()
    result = optimize(current_load_kw=735, predicted_peak_kw=824, required_reduction_kw=74,
                       state=state, objective="COST_FIRST")
    selected_ids = {a.equipment_id for a in result.actions}
    assert "CONTROL_ROOM_HVAC" not in selected_ids
    assert "PRODUCTION_LINE_1" not in selected_ids


def test_optimizer_avoids_opted_out_zone():
    state = standard_state(opted_out_zones={"WAREHOUSE"})
    result = optimize(current_load_kw=735, predicted_peak_kw=824, required_reduction_kw=50,
                       state=state, objective="COST_FIRST")
    selected_zones = {a.zone_id for a in result.actions}
    assert "WAREHOUSE" not in selected_zones


def test_optimizer_excludes_maintenance_equipment():
    state = standard_state(equipment_status={"COMPRESSOR_B": "MAINTENANCE"})
    result = optimize(current_load_kw=735, predicted_peak_kw=824, required_reduction_kw=74,
                       state=state, objective="COST_FIRST")
    selected_ids = {a.equipment_id for a in result.actions}
    assert "COMPRESSOR_B" not in selected_ids
    assert ("COMPRESSOR_B", ["equipment_under_maintenance"]) in result.rejected_candidates


def test_occupant_first_produces_lower_or_equal_comfort_risk_than_cost_first():
    state = standard_state()
    cost_first = optimize(current_load_kw=735, predicted_peak_kw=824, required_reduction_kw=74,
                           state=state, objective="COST_FIRST")
    occupant_first = optimize(current_load_kw=735, predicted_peak_kw=824, required_reduction_kw=74,
                               state=state, objective="OCCUPANT_FIRST")
    assert occupant_first.comfort_risk_score <= cost_first.comfort_risk_score + 1e-6


def test_no_feasible_plan_when_everything_blocked():
    state = standard_state(
        equipment_status={"COMPRESSOR_A": "MAINTENANCE", "COMPRESSOR_B": "MAINTENANCE"},
        opted_out_zones={"WAREHOUSE", "OFFICE_A", "UTILITY", "BATTERY_FLEX"},
    )
    result = optimize(current_load_kw=900, predicted_peak_kw=1000, required_reduction_kw=250,
                       state=state, objective="COST_FIRST")
    assert result.feasible is False
    assert result.shortfall_kw == 250


def test_extreme_heat_avoids_hvac_and_shifts_to_other_equipment():
    hot_state = standard_state(zone_temperatures={
        "OFFICE_A": 37, "WAREHOUSE": 39, "PRODUCTION": 38,
        "CONTROL_ROOM": 22, "UTILITY": 40, "BATTERY_FLEX": 38,
    })
    result = optimize(current_load_kw=735, predicted_peak_kw=824, required_reduction_kw=74,
                       state=hot_state, objective="COST_FIRST")
    hvac_actions = [a for a in result.actions if a.equipment_type == "HVAC"]
    assert len(hvac_actions) == 0 or sum(a.reduction_kw for a in hvac_actions) < 5
