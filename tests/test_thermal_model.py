"""
Unit tests for the 1R1C and 2R2C lumped-capacitance thermal models
and physics-informed comfort guarantee verification.
"""
import pytest
from planner.thermal_model import (
    simulate_1r1c,
    simulate_2r2c,
    projected_temperature_rise,
    max_safe_reduction,
    get_thermal_params,
    ZoneThermalParams,
)


def test_thermal_params_loading():
    params = get_thermal_params("OFFICE_A")
    assert params.zone_id == "OFFICE_A"
    assert params.R_env > 0
    assert params.C_air > 0
    assert params.C_wall > 0
    assert params.hvac_capacity_kw > 0


def test_1r1c_simulation_temperature_dynamics():
    params = get_thermal_params("OFFICE_A", T_outdoor=35.0)
    # 0 kW reduction: HVAC at full cooling capacity keeps temperature stable or cooling
    res_zero = simulate_1r1c(T_initial=22.0, params=params, hvac_reduction_kw=0.0, duration_minutes=30.0)
    assert res_zero.T_final <= 23.0

    # Large reduction: temperature must rise
    res_curtailed = simulate_1r1c(T_initial=22.0, params=params, hvac_reduction_kw=25.0, duration_minutes=30.0)
    assert res_curtailed.T_final > res_zero.T_final
    assert len(res_curtailed.T_trajectory) == 31


def test_2r2c_wall_buffering_effect():
    """
    Thermal mass test: 2R2C model should exhibit thermal buffering from wall mass,
    producing a slower initial rate of temperature rise than the unbuffered 1R1C model.
    """
    params = get_thermal_params("PRODUCTION", T_outdoor=38.0)
    T_init = 23.0
    reduction_kw = 30.0

    res_1r1c = simulate_1r1c(T_init, params, reduction_kw, duration_minutes=30.0)
    res_2r2c = simulate_2r2c(T_init, None, params, reduction_kw, duration_minutes=30.0)

    # Wall thermal mass absorbs part of the heat: 2R2C final temperature should be lower
    assert res_2r2c.T_final <= res_1r1c.T_final


def test_projected_temperature_rise_limits():
    pred = projected_temperature_rise(
        zone_id="OFFICE_A",
        current_temp=23.5,
        hvac_reduction_kw=20.0,
        duration_minutes=30.0,
        comfort_max=24.0,
        safety_limit=27.0,
        model="2R2C",
    )
    assert pred.model_type == "2R2C"
    assert pred.T_max >= pred.T_initial
    assert pred.total_minutes == 30.0
    if pred.T_max > 24.0:
        assert pred.time_to_comfort_limit is not None


def test_max_safe_reduction_binary_search():
    # If starting at 22°C with comfort limit 24°C and safety margin 3.0°C (limit 27°C),
    # there should be significant safe curtailment capacity available
    safe_kw = max_safe_reduction(
        zone_id="OFFICE_A",
        current_temp=22.0,
        comfort_max=24.0,
        safety_margin=3.0,
        duration_minutes=30.0,
        model="2R2C",
    )
    assert safe_kw > 5.0

    # If starting already at the absolute limit (27°C), safe reduction must be 0
    safe_kw_hot = max_safe_reduction(
        zone_id="OFFICE_A",
        current_temp=27.0,
        comfort_max=24.0,
        safety_margin=3.0,
        duration_minutes=30.0,
        model="2R2C",
    )
    assert safe_kw_hot == 0.0
