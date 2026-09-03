import os
import pytest
from pathlib import Path

TEST_DB = Path(__file__).resolve().parent.parent / "data" / "processed" / "test_oadrp.db"


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    import app.database as db_mod
    if TEST_DB.exists():
        TEST_DB.unlink()
    monkeypatch.setattr(db_mod, "DB_PATH", TEST_DB)
    yield
    if TEST_DB.exists():
        TEST_DB.unlink()


def test_online_observation_marked_synced():
    from app.database import save_field_observation
    result = save_field_observation({"zone_id": "OFFICE_A", "occupancy_band": "MEDIUM"}, online=True)
    assert result["sync_state"] == "SYNCED"


def test_offline_observation_queued():
    from app.database import save_field_observation
    result = save_field_observation({"zone_id": "OFFICE_A", "occupancy_band": "MEDIUM"}, online=False)
    assert result["sync_state"] == "QUEUED"


def test_conflict_detected_when_equipment_status_disagrees():
    from app.database import set_equipment_status, save_field_observation
    set_equipment_status("COMPRESSOR_B", "MAINTENANCE")
    result = save_field_observation(
        {"zone_id": "PRODUCTION", "equipment_id": "COMPRESSOR_B", "equipment_status": "AVAILABLE"},
        online=False,
    )
    assert result["sync_state"] == "CONFLICT"
    assert "MAINTENANCE" in result["conflict_details"]


def test_sync_queued_does_not_clear_conflicts():
    from app.database import set_equipment_status, save_field_observation, sync_queued_observations
    set_equipment_status("COMPRESSOR_B", "MAINTENANCE")
    save_field_observation({"zone_id": "PRODUCTION", "equipment_id": "COMPRESSOR_B",
                              "equipment_status": "AVAILABLE"}, online=False)
    save_field_observation({"zone_id": "OFFICE_A", "occupancy_band": "LOW"}, online=False)
    result = sync_queued_observations()
    assert result["synced"] == 1
    assert result["remaining_conflicts"] == 1


def test_opt_out_roundtrip():
    from app.database import save_opt_out, get_active_opt_outs
    save_opt_out("WAREHOUSE", "TEMPORARY_OPT_OUT", "Cleaning crew", "13:00", "16:00")
    assert "WAREHOUSE" in get_active_opt_outs()


def test_override_audit_log_records_entry():
    from app.database import save_override, get_overrides
    save_override("mgr1", "Facility Manager", "Grid emergency", "Preferred comfort", 30, "Extend HVAC reduction")
    overrides = get_overrides()
    assert len(overrides) == 1
    assert overrides[0]["authorized_user"] == "mgr1"
