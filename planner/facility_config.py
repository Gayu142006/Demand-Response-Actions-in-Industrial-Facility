"""
Facility configuration: zones, comfort ranges, equipment, and tariff.

This is the single source of truth for the synthetic facility used across
data generation, the planner, the app, and the evaluation experiment.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Zone:
    zone_id: str
    zone_type: str          # PRODUCTION, WAREHOUSE, OFFICE, CONTROL_ROOM, UTILITY, BATTERY
    comfort_min: float      # deg C, absolute safety floor is comfort_min - safety_margin
    comfort_max: float
    criticality: str        # CRITICAL, HIGH, MEDIUM, LOW
    base_load_kw: float     # nominal load contribution at typical occupancy


@dataclass
class Equipment:
    equipment_id: str
    zone_id: str
    equipment_type: str     # HVAC, COMPRESSOR, PUMP, LIGHTING, BATTERY, PRODUCTION_LINE
    max_reduction_kw: float
    min_reduction_kw: float
    flexibility: str        # HIGH, MEDIUM, LOW
    is_critical: bool = False
    max_duration_min: int = 60


# --- Zones -----------------------------------------------------------------
ZONES = [
    Zone("PRODUCTION", "PRODUCTION", comfort_min=20, comfort_max=26, criticality="CRITICAL", base_load_kw=260),
    Zone("WAREHOUSE", "WAREHOUSE", comfort_min=18, comfort_max=28, criticality="MEDIUM", base_load_kw=90),
    Zone("OFFICE_A", "OFFICE", comfort_min=21, comfort_max=24, criticality="LOW", base_load_kw=55),
    Zone("CONTROL_ROOM", "CONTROL_ROOM", comfort_min=20, comfort_max=23, criticality="CRITICAL", base_load_kw=35),
    Zone("UTILITY", "UTILITY", comfort_min=15, comfort_max=32, criticality="MEDIUM", base_load_kw=70),
    Zone("BATTERY_FLEX", "BATTERY", comfort_min=10, comfort_max=35, criticality="LOW", base_load_kw=60),
]

ZONE_SAFETY_MARGIN = 3.0  # deg C beyond comfort range before "absolute" safety limit

# --- Equipment / candidate demand-response actions --------------------------
EQUIPMENT = [
    Equipment("HVAC_OFFICE", "OFFICE_A", "HVAC", max_reduction_kw=35, min_reduction_kw=10, flexibility="MEDIUM"),
    Equipment("HVAC_WAREHOUSE", "WAREHOUSE", "HVAC", max_reduction_kw=28, min_reduction_kw=8, flexibility="HIGH"),
    Equipment("COMPRESSOR_A", "PRODUCTION", "COMPRESSOR", max_reduction_kw=45, min_reduction_kw=15, flexibility="HIGH"),
    Equipment("COMPRESSOR_B", "PRODUCTION", "COMPRESSOR", max_reduction_kw=40, min_reduction_kw=15, flexibility="HIGH"),
    Equipment("PUMP_FLEX_1", "UTILITY", "PUMP", max_reduction_kw=22, min_reduction_kw=8, flexibility="HIGH"),
    Equipment("BATTERY_CHARGE", "BATTERY_FLEX", "BATTERY", max_reduction_kw=90, min_reduction_kw=20, flexibility="HIGH"),
    Equipment("LIGHTING_WAREHOUSE", "WAREHOUSE", "LIGHTING", max_reduction_kw=12, min_reduction_kw=3, flexibility="MEDIUM"),
    Equipment("PRODUCTION_LINE_1", "PRODUCTION", "PRODUCTION_LINE", max_reduction_kw=30, min_reduction_kw=0,
              flexibility="LOW", is_critical=True),
    Equipment("CONTROL_ROOM_HVAC", "CONTROL_ROOM", "HVAC", max_reduction_kw=0, min_reduction_kw=0,
              flexibility="LOW", is_critical=True),  # never flexible - life-safety adjacent
]

# --- Tariff ------------------------------------------------------------------
PEAK_THRESHOLD_KW = 750.0
DEMAND_RATE_PER_KW = 60.0     # currency units per kW of peak demand (INR-style, per spec)
ENERGY_TARIFF_OFFPEAK = 6.5   # per kWh, 00:00-08:00 and 21:00-24:00
ENERGY_TARIFF_MID = 8.5       # per kWh, 08:00-17:00
ENERGY_TARIFF_PEAK = 11.0     # per kWh, 17:00-21:00

# --- Operational hard limits ---------------------------------------------
MINIMUM_PRODUCTION_LOAD_KW = 180.0   # production must never be optimized below this


def energy_tariff_for_hour(hour: int) -> float:
    if 8 <= hour < 17:
        return ENERGY_TARIFF_MID
    if 17 <= hour < 21:
        return ENERGY_TARIFF_PEAK
    return ENERGY_TARIFF_OFFPEAK


def get_zone(zone_id: str) -> Zone:
    for z in ZONES:
        if z.zone_id == zone_id:
            return z
    raise KeyError(zone_id)


def equipment_for_zone(zone_id: str):
    return [e for e in EQUIPMENT if e.zone_id == zone_id]
