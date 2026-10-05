"""
Unit tests for new features:
  - Equipment cycling and fatigue tracking engine
  - Adaptive feedback-driven planner tuner
  - REST API endpoint handlers
"""
import pytest
import asyncio
from pathlib import Path
from starlette.requests import Request

from planner.equipment_fatigue import EquipmentFatigueTracker
from validation.adaptive_tuner import (
    PlannerCalibrationProfile,
    compute_calibration_from_feedback,
    save_calibration,
    load_calibration,
)
from app.api import health, thermal_headroom_endpoint, equipment_fatigue_endpoint


def test_equipment_fatigue_tracker(tmp_path):
    p = tmp_path / "fatigue.json"
    tracker = EquipmentFatigueTracker(persistence_file=p)

    rec = tracker.get_or_create("COMPRESSOR_A", "COMPRESSOR")
    assert rec.fatigue_index == 0.0

    # Record 5 curtailments (0.5 hrs each, 20 kW)
    for _ in range(5):
        tracker.record_curtailment("COMPRESSOR_A", "COMPRESSOR", reduction_kw=20.0, duration_hours=0.5)

    updated_rec = tracker.records["COMPRESSOR_A"]
    assert updated_rec.total_events == 5
    assert updated_rec.total_curtailed_kwh == 50.0
    assert updated_rec.fatigue_index > 0.0

    # Ensure persistence works
    tracker2 = EquipmentFatigueTracker(persistence_file=p)
    assert tracker2.records["COMPRESSOR_A"].total_events == 5


def test_adaptive_calibration():
    profile = compute_calibration_from_feedback()
    assert profile.comfort_weight_occ_first >= 8.0
    assert profile.safety_margin_c >= 3.0

    # Test save/load
    profile.safety_margin_c = 3.5
    save_calibration(profile)
    loaded = load_calibration()
    assert loaded.safety_margin_c == 3.5


def test_rest_api_endpoints():
    async def _run():
        req = Request({"type": "http", "method": "GET", "path": "/api/v1/health", "headers": []})
        resp = await health(req)
        assert resp.status_code == 200

        req_hr = Request({
            "type": "http",
            "method": "GET",
            "path": "/api/v1/thermal/headroom",
            "query_string": b"zone_id=OFFICE_A&current_temp=22.0&comfort_max=24.0",
            "headers": [],
        })
        resp_hr = await thermal_headroom_endpoint(req_hr)
        assert resp_hr.status_code == 200

        req_fatigue = Request({
            "type": "http",
            "method": "GET",
            "path": "/api/v1/equipment/fatigue",
            "headers": [],
        })
        resp_fatigue = await equipment_fatigue_endpoint(req_fatigue)
        assert resp_fatigue.status_code == 200

    asyncio.run(_run())
