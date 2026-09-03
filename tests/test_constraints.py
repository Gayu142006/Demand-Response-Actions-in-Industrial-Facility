import pytest
from planner.constraints import ActionCandidate, FacilityState, validate_hard_constraints
from planner.facility_config import MINIMUM_PRODUCTION_LOAD_KW


def make_state(**overrides):
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


def test_critical_equipment_cannot_be_reduced():
    cand = ActionCandidate("CONTROL_ROOM_HVAC", "CONTROL_ROOM", "HVAC", 10, is_critical=True, flexibility="LOW")
    result = validate_hard_constraints(cand, make_state())
    assert not result.passed
    assert "critical_equipment_must_remain_on" in result.violated_rules


def test_maintenance_equipment_rejected():
    cand = ActionCandidate("COMPRESSOR_B", "PRODUCTION", "COMPRESSOR", 40, is_critical=False, flexibility="HIGH")
    state = make_state(equipment_status={"COMPRESSOR_B": "MAINTENANCE"})
    result = validate_hard_constraints(cand, state)
    assert not result.passed
    assert "equipment_under_maintenance" in result.violated_rules


def test_opted_out_zone_rejected():
    cand = ActionCandidate("HVAC_WAREHOUSE", "WAREHOUSE", "HVAC", 20, is_critical=False, flexibility="HIGH")
    state = make_state(opted_out_zones={"WAREHOUSE"})
    result = validate_hard_constraints(cand, state)
    assert not result.passed
    assert "zone_opted_out" in result.violated_rules


def test_production_cannot_go_below_minimum():
    # Test 1 from spec section 59: production min=500 (facility default lower,
    # so use current_production_load_kw explicitly here)
    cand = ActionCandidate("PRODUCTION_LINE_1", "PRODUCTION", "PRODUCTION_LINE", 30,
                            is_critical=False, flexibility="LOW")
    state = make_state(current_production_load_kw=MINIMUM_PRODUCTION_LOAD_KW + 10)
    result = validate_hard_constraints(cand, state)
    assert not result.passed
    assert "production_below_minimum" in result.violated_rules


def test_hvac_within_safe_temperature_passes():
    cand = ActionCandidate("HVAC_OFFICE", "OFFICE_A", "HVAC", 5, is_critical=False, flexibility="MEDIUM")
    result = validate_hard_constraints(cand, make_state())
    assert result.passed


def test_hvac_extreme_reduction_violates_absolute_safety_limit():
    # OFFICE_A comfort_max=24, safety margin=3 -> absolute max 27.
    # Starting at 26.9C, a large reduction pushes projected temp over 27C.
    cand = ActionCandidate("HVAC_OFFICE", "OFFICE_A", "HVAC", 35, is_critical=False, flexibility="MEDIUM")
    state = make_state(zone_temperatures={"OFFICE_A": 26.9, "WAREHOUSE": 24, "PRODUCTION": 23,
                                            "CONTROL_ROOM": 21, "UTILITY": 25, "BATTERY_FLEX": 22})
    result = validate_hard_constraints(cand, state)
    assert not result.passed
    assert "temperature_absolute_safety_limit" in result.violated_rules


def test_missing_temperature_blocks_large_hvac_reduction():
    cand = ActionCandidate("HVAC_OFFICE", "OFFICE_A", "HVAC", 30, is_critical=False, flexibility="MEDIUM")
    state = make_state(zone_temperatures={"OFFICE_A": None, "WAREHOUSE": 24, "PRODUCTION": 23,
                                            "CONTROL_ROOM": 21, "UTILITY": 25, "BATTERY_FLEX": 22})
    result = validate_hard_constraints(cand, state)
    assert not result.passed
    assert "unknown_temperature_conservative_block" in result.violated_rules


def test_negative_reduction_invalid():
    cand = ActionCandidate("PUMP_FLEX_1", "UTILITY", "PUMP", -5, is_critical=False, flexibility="HIGH")
    result = validate_hard_constraints(cand, make_state())
    assert not result.passed
    assert "negative_reduction_invalid" in result.violated_rules
