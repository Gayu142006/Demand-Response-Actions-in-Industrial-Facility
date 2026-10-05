"""
Lumped-Capacitance Thermal Model (1R1C and 2R2C).

Replaces the linear approximation ``projected_rise = reduction_kw * 0.06``
with physics-informed models that account for:
  - Thermal mass (capacitance) of the zone
  - Wall/envelope thermal resistance
  - Internal heat gains (occupancy, equipment)
  - Outdoor temperature driving force
  - HVAC cooling capacity and its reduction during a DR event

Two models are provided:
  1R1C: Single-node model (one R, one C). Fast, suitable for open-plan
        zones where air and surfaces are approximately co-located.
  2R2C: Two-node model (air + wall mass). More accurate for zones with
        significant thermal mass (concrete, masonry) that buffers
        temperature swings during short DR events.

Both produce a predicted zone temperature trajectory over a DR event
duration, which the constraint engine and optimizer use to validate
comfort guarantees more accurately than the linear approximation.

Reference: ISO 13790 / EN ISO 52016, simplified RC-network models.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ZoneThermalParams:
    """Physical parameters for a zone's thermal model."""
    zone_id: str

    # Thermal resistance (envelope) [°C/kW] — higher = better insulated
    R_env: float = 8.0           # typical industrial zone: ~5-15 °C/kW

    # Thermal capacitance (zone air + lightweight contents) [kWh/°C]
    C_air: float = 2.5           # typical: 1-5 kWh/°C for industrial zones

    # For 2R2C: wall/slab thermal mass
    R_wall: float = 3.0          # internal wall resistance [°C/kW]
    C_wall: float = 15.0         # wall thermal mass [kWh/°C] — concrete/masonry much higher

    # HVAC parameters
    hvac_capacity_kw: float = 35.0    # total cooling capacity [kW thermal]
    hvac_cop: float = 3.5             # coefficient of performance

    # Internal gains baseline [kW thermal]
    internal_gains_kw: float = 5.0    # occupancy + equipment heat rejection

    # Outdoor temperature (updated per timestep)
    T_outdoor: float = 35.0


# Default thermal parameters per zone type
DEFAULT_THERMAL_PARAMS = {
    "PRODUCTION":    ZoneThermalParams("PRODUCTION", R_env=5.0, C_air=4.0, R_wall=2.5, C_wall=25.0,
                                        hvac_capacity_kw=50.0, hvac_cop=3.0, internal_gains_kw=15.0, T_outdoor=35.0),
    "WAREHOUSE":     ZoneThermalParams("WAREHOUSE", R_env=6.0, C_air=5.0, R_wall=3.0, C_wall=20.0,
                                        hvac_capacity_kw=40.0, hvac_cop=3.5, internal_gains_kw=5.0, T_outdoor=35.0),
    "OFFICE_A":      ZoneThermalParams("OFFICE_A", R_env=10.0, C_air=2.0, R_wall=4.0, C_wall=12.0,
                                        hvac_capacity_kw=35.0, hvac_cop=4.0, internal_gains_kw=8.0, T_outdoor=35.0),
    "CONTROL_ROOM":  ZoneThermalParams("CONTROL_ROOM", R_env=12.0, C_air=1.5, R_wall=5.0, C_wall=10.0,
                                        hvac_capacity_kw=20.0, hvac_cop=4.0, internal_gains_kw=10.0, T_outdoor=32.0),
    "UTILITY":       ZoneThermalParams("UTILITY", R_env=4.0, C_air=3.0, R_wall=2.0, C_wall=18.0,
                                        hvac_capacity_kw=30.0, hvac_cop=3.0, internal_gains_kw=12.0, T_outdoor=35.0),
    "BATTERY_FLEX":  ZoneThermalParams("BATTERY_FLEX", R_env=6.0, C_air=2.5, R_wall=3.0, C_wall=15.0,
                                        hvac_capacity_kw=25.0, hvac_cop=3.5, internal_gains_kw=3.0, T_outdoor=35.0),
}


def get_thermal_params(zone_id: str, T_outdoor: Optional[float] = None) -> ZoneThermalParams:
    """Get thermal parameters for a zone, with optional outdoor temp override."""
    params = DEFAULT_THERMAL_PARAMS.get(zone_id)
    if params is None:
        # Fallback: use conservative defaults
        params = ZoneThermalParams(zone_id)
    if T_outdoor is not None:
        params = ZoneThermalParams(
            zone_id=params.zone_id, R_env=params.R_env, C_air=params.C_air,
            R_wall=params.R_wall, C_wall=params.C_wall,
            hvac_capacity_kw=params.hvac_capacity_kw, hvac_cop=params.hvac_cop,
            internal_gains_kw=params.internal_gains_kw, T_outdoor=T_outdoor,
        )
    return params


# ---------------------------------------------------------------------------
# 1R1C Model
# ---------------------------------------------------------------------------

@dataclass
class ThermalPrediction:
    """Result of a thermal model simulation."""
    T_initial: float
    T_final: float
    T_max: float
    T_trajectory: List[float]         # temperature at each timestep
    time_to_comfort_limit: Optional[float]  # minutes until comfort_max exceeded (None if never)
    time_to_safety_limit: Optional[float]   # minutes until safety limit exceeded (None if never)
    model_type: str                   # "1R1C" | "2R2C"
    dt_minutes: float
    total_minutes: float


def simulate_1r1c(
    T_initial: float,
    params: ZoneThermalParams,
    hvac_reduction_kw: float,
    duration_minutes: float = 60.0,
    dt_minutes: float = 1.0,
    comfort_max: float = 24.0,
    safety_limit: float = 27.0,
) -> ThermalPrediction:
    """
    1R1C (single-node) lumped-capacitance thermal model.

    Energy balance for zone air node:
        C_air * dT/dt = Q_gains - Q_hvac + (T_outdoor - T) / R_env

    where:
        Q_gains = internal_gains_kw (occupancy + equipment heat)
        Q_hvac  = hvac_capacity_kw * COP_adj - hvac_reduction_kw * COP_adj
                  (reduced cooling capacity during DR event)
        (T_outdoor - T) / R_env = heat flow through envelope

    Discretized with forward Euler:
        T(t+dt) = T(t) + (dt/C_air) * [Q_gains - Q_hvac_net + (T_out - T(t))/R_env]
    """
    dt_hours = dt_minutes / 60.0
    n_steps = int(duration_minutes / dt_minutes)

    # Effective cooling power AFTER DR reduction
    # hvac_reduction_kw is in electrical kW; thermal effect is scaled by COP
    hvac_thermal_full = params.hvac_capacity_kw  # total thermal cooling capacity
    hvac_reduction_thermal = hvac_reduction_kw * params.hvac_cop
    hvac_thermal_net = max(0.0, hvac_thermal_full - hvac_reduction_thermal)

    T = T_initial
    trajectory = [T]
    T_max = T
    time_comfort = None
    time_safety = None

    for step in range(n_steps):
        t_minutes = (step + 1) * dt_minutes

        # Heat flows [kW thermal]:
        Q_gains = params.internal_gains_kw
        Q_envelope = (params.T_outdoor - T) / params.R_env  # positive when outdoor > indoor
        Q_net = Q_gains + Q_envelope - hvac_thermal_net      # net heat into zone

        # Temperature update
        dT = (dt_hours / params.C_air) * Q_net
        T = T + dT

        trajectory.append(T)
        T_max = max(T_max, T)

        if time_comfort is None and T > comfort_max:
            time_comfort = t_minutes
        if time_safety is None and T > safety_limit:
            time_safety = t_minutes

    return ThermalPrediction(
        T_initial=T_initial,
        T_final=round(T, 3),
        T_max=round(T_max, 3),
        T_trajectory=[round(t, 3) for t in trajectory],
        time_to_comfort_limit=time_comfort,
        time_to_safety_limit=time_safety,
        model_type="1R1C",
        dt_minutes=dt_minutes,
        total_minutes=duration_minutes,
    )


# ---------------------------------------------------------------------------
# 2R2C Model
# ---------------------------------------------------------------------------

def simulate_2r2c(
    T_air_initial: float,
    T_wall_initial: Optional[float],
    params: ZoneThermalParams,
    hvac_reduction_kw: float,
    duration_minutes: float = 60.0,
    dt_minutes: float = 1.0,
    comfort_max: float = 24.0,
    safety_limit: float = 27.0,
) -> ThermalPrediction:
    """
    2R2C (two-node) lumped-capacitance thermal model.

    Two coupled nodes: air temperature and wall/slab temperature.

    Air node:
        C_air * dT_air/dt = Q_gains - Q_hvac + (T_wall - T_air) / R_wall
                            + (T_outdoor - T_air) / R_env (infiltration/windows)

    Wall node:
        C_wall * dT_wall/dt = (T_outdoor - T_wall) / R_env + (T_air - T_wall) / R_wall

    The 2R2C model captures the buffering effect of thermal mass: during a
    short DR event (30-60 min), a zone with heavy walls will warm up much
    more slowly than the 1R1C model predicts, because the wall mass absorbs
    the excess heat. This makes comfort guarantees LESS conservative (and
    more accurate) for thermally massive zones.
    """
    dt_hours = dt_minutes / 60.0
    n_steps = int(duration_minutes / dt_minutes)

    # If wall temperature not known, assume it's near the air temperature
    if T_wall_initial is None:
        T_wall_initial = T_air_initial + 0.5

    T_air = T_air_initial
    T_wall = T_wall_initial

    hvac_thermal_full = params.hvac_capacity_kw
    hvac_reduction_thermal = hvac_reduction_kw * params.hvac_cop
    hvac_thermal_net = max(0.0, hvac_thermal_full - hvac_reduction_thermal)

    trajectory = [T_air]
    T_max = T_air
    time_comfort = None
    time_safety = None

    for step in range(n_steps):
        t_minutes = (step + 1) * dt_minutes

        # Air node heat flows
        Q_gains = params.internal_gains_kw
        Q_infil = (params.T_outdoor - T_air) / (params.R_env * 2)  # infiltration: half of envelope R
        Q_wall_to_air = (T_wall - T_air) / params.R_wall
        Q_air_net = Q_gains + Q_infil + Q_wall_to_air - hvac_thermal_net

        # Wall node heat flows
        Q_env_to_wall = (params.T_outdoor - T_wall) / (params.R_env * 2)  # other half of envelope R
        Q_air_to_wall = (T_air - T_wall) / params.R_wall
        Q_wall_net = Q_env_to_wall + Q_air_to_wall

        # Temperature updates (forward Euler)
        dT_air = (dt_hours / params.C_air) * Q_air_net
        dT_wall = (dt_hours / params.C_wall) * Q_wall_net

        T_air = T_air + dT_air
        T_wall = T_wall + dT_wall

        trajectory.append(T_air)
        T_max = max(T_max, T_air)

        if time_comfort is None and T_air > comfort_max:
            time_comfort = t_minutes
        if time_safety is None and T_air > safety_limit:
            time_safety = t_minutes

    return ThermalPrediction(
        T_initial=T_air_initial,
        T_final=round(T_air, 3),
        T_max=round(T_max, 3),
        T_trajectory=[round(t, 3) for t in trajectory],
        time_to_comfort_limit=time_comfort,
        time_to_safety_limit=time_safety,
        model_type="2R2C",
        dt_minutes=dt_minutes,
        total_minutes=duration_minutes,
    )


# ---------------------------------------------------------------------------
# Interface for the constraint engine
# ---------------------------------------------------------------------------

def projected_temperature_rise(
    zone_id: str,
    current_temp: float,
    hvac_reduction_kw: float,
    duration_minutes: float = 30.0,
    comfort_max: float = 24.0,
    safety_limit: float = 27.0,
    T_outdoor: Optional[float] = None,
    model: str = "2R2C",
) -> ThermalPrediction:
    """
    Main interface for the constraint engine and optimizer.

    Replaces the old linear approximation:
        projected_rise = reduction_kw * 0.06

    with a physics-informed prediction based on the zone's thermal properties.

    Args:
        zone_id: Facility zone identifier
        current_temp: Current zone air temperature [°C]
        hvac_reduction_kw: Electrical kW reduction to HVAC system
        duration_minutes: DR event duration
        comfort_max: Comfort ceiling [°C]
        safety_limit: Absolute safety limit [°C]
        T_outdoor: Current outdoor temperature (None = use default)
        model: "1R1C" or "2R2C"

    Returns:
        ThermalPrediction with full trajectory and limit-crossing times
    """
    params = get_thermal_params(zone_id, T_outdoor)

    if model == "1R1C":
        return simulate_1r1c(
            T_initial=current_temp,
            params=params,
            hvac_reduction_kw=hvac_reduction_kw,
            duration_minutes=duration_minutes,
            comfort_max=comfort_max,
            safety_limit=safety_limit,
        )
    else:
        return simulate_2r2c(
            T_air_initial=current_temp,
            T_wall_initial=None,  # estimated internally
            params=params,
            hvac_reduction_kw=hvac_reduction_kw,
            duration_minutes=duration_minutes,
            comfort_max=comfort_max,
            safety_limit=safety_limit,
        )


def max_safe_reduction(
    zone_id: str,
    current_temp: float,
    comfort_max: float,
    safety_margin: float = 3.0,
    duration_minutes: float = 30.0,
    T_outdoor: Optional[float] = None,
    model: str = "2R2C",
) -> float:
    """
    Binary-search for the maximum HVAC reduction [electrical kW] that keeps
    the zone temperature below the absolute safety limit (comfort_max +
    safety_margin) for the entire DR event duration.

    This replaces the old linear headroom calculation:
        headroom_kw = max(0, headroom_c / 0.06)
    """
    safety_limit = comfort_max + safety_margin
    if current_temp >= safety_limit:
        return 0.0

    lo, hi = 0.0, 100.0  # search range [kW]
    best = 0.0

    for _ in range(20):  # ~1e-6 precision in 20 iterations
        mid = (lo + hi) / 2.0
        pred = projected_temperature_rise(
            zone_id=zone_id,
            current_temp=current_temp,
            hvac_reduction_kw=mid,
            duration_minutes=duration_minutes,
            comfort_max=comfort_max,
            safety_limit=safety_limit,
            T_outdoor=T_outdoor,
            model=model,
        )
        if pred.T_max <= safety_limit:
            best = mid
            lo = mid
        else:
            hi = mid

    return round(best, 2)
