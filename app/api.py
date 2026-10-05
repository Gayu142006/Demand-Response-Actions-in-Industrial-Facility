"""
REST API Backend (Starlette / ASGI).

Provides HTTP endpoints for external BMS integrations, telemetry ingestion,
thermal modeling, and stakeholder validation.
"""
import json
from dataclasses import asdict
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.requests import Request

from planner.planner_engine import run_planner
from planner.facility_config import PEAK_THRESHOLD_KW
from planner.thermal_model import projected_temperature_rise, max_safe_reduction, DEFAULT_THERMAL_PARAMS
from planner.iot_gateway import (
    EdgeGateway,
    create_example_bacnet_mappings,
    create_example_modbus_mappings,
    create_example_mqtt_mappings,
    BACnetAdapter,
    ModbusAdapter,
    MQTTAdapter,
)
from planner.equipment_fatigue import EquipmentFatigueTracker
from validation.protocol import (
    ALL_QUESTIONS,
    ValidationSession,
    QuestionResponse,
    save_session,
    load_all_sessions,
    aggregate_feedback_report,
    generate_design_recommendations,
)
from validation.adaptive_tuner import compute_calibration_from_feedback, load_calibration

# Singleton instances
gateway = EdgeGateway()
fatigue_tracker = EquipmentFatigueTracker()


async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "service": "oadrp-api",
        "version": "2.0.0",
        "peak_threshold_kw": PEAK_THRESHOLD_KW,
    })


async def plan_endpoint(request: Request) -> JSONResponse:
    data = await request.json()
    current_load_kw = float(data.get("current_load_kw", 750.0))
    predicted_peak_kw = float(data.get("predicted_peak_kw", 820.0))
    objective = data.get("objective", "COST_FIRST")
    zone_temps = data.get("zone_temperatures", {})
    equip_status = data.get("equipment_status", {})
    opted_out = set(data.get("opted_out_zones", []))
    pin_zones = set(data.get("participate_if_necessary_zones", []))
    prod_load = float(data.get("current_production_load_kw", 250.0))

    out = run_planner(
        current_load_kw=current_load_kw,
        predicted_peak_kw=predicted_peak_kw,
        zone_temperatures=zone_temps,
        equipment_status=equip_status,
        opted_out_zones=opted_out,
        current_production_load_kw=prod_load,
        objective=objective,
        participate_if_necessary_zones=pin_zones,
    )

    plan_dict = {
        "feasible": out.plan.feasible,
        "total_reduction_kw": out.plan.total_reduction_kw,
        "planned_peak_kw": out.plan.planned_peak_kw,
        "shortfall_kw": out.plan.shortfall_kw,
        "comfort_risk_score": out.plan.comfort_risk_score,
        "actions": [asdict(a) for a in out.plan.actions],
        "validation": asdict(out.validation),
    }
    return JSONResponse(plan_dict)


async def thermal_simulate_endpoint(request: Request) -> JSONResponse:
    data = await request.json()
    zone_id = data.get("zone_id", "OFFICE_A")
    current_temp = float(data.get("current_temp", 23.0))
    reduction_kw = float(data.get("hvac_reduction_kw", 15.0))
    duration_min = float(data.get("duration_minutes", 30.0))
    model = data.get("model", "2R2C")

    pred = projected_temperature_rise(
        zone_id=zone_id,
        current_temp=current_temp,
        hvac_reduction_kw=reduction_kw,
        duration_minutes=duration_min,
        model=model,
    )
    return JSONResponse(asdict(pred))


async def thermal_headroom_endpoint(request: Request) -> JSONResponse:
    params = request.query_params
    zone_id = params.get("zone_id", "OFFICE_A")
    current_temp = float(params.get("current_temp", 23.0))
    comfort_max = float(params.get("comfort_max", 24.0))

    headroom_kw = max_safe_reduction(
        zone_id=zone_id,
        current_temp=current_temp,
        comfort_max=comfort_max,
    )
    return JSONResponse({
        "zone_id": zone_id,
        "current_temp": current_temp,
        "max_safe_reduction_kw": headroom_kw,
    })


async def gateway_health_endpoint(request: Request) -> JSONResponse:
    return JSONResponse(gateway.get_health_status())


async def equipment_fatigue_endpoint(request: Request) -> JSONResponse:
    return JSONResponse({"fleet": fatigue_tracker.get_fleet_status()})


async def validation_report_endpoint(request: Request) -> JSONResponse:
    sessions = load_all_sessions()
    report = aggregate_feedback_report(sessions)
    recs = generate_design_recommendations(report)
    return JSONResponse({
        "report": report,
        "design_recommendations": recs,
        "active_calibration": asdict(load_calibration()),
    })


routes = [
    Route("/api/v1/health", health, methods=["GET"]),
    Route("/api/v1/plan", plan_endpoint, methods=["POST"]),
    Route("/api/v1/thermal/simulate", thermal_simulate_endpoint, methods=["POST"]),
    Route("/api/v1/thermal/headroom", thermal_headroom_endpoint, methods=["GET"]),
    Route("/api/v1/gateway/health", gateway_health_endpoint, methods=["GET"]),
    Route("/api/v1/equipment/fatigue", equipment_fatigue_endpoint, methods=["GET"]),
    Route("/api/v1/validation/report", validation_report_endpoint, methods=["GET"]),
]

app = Starlette(debug=True, routes=routes)
