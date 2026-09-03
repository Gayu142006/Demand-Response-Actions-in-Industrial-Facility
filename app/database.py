"""
SQLite-backed storage for:
  - field observations (with offline/online sync state)
  - opt-out preferences
  - authorized overrides (with audit log)
  - approved plans

Spec sections 29-32 describe the offline/low-bandwidth workflow: field
observations are always written locally first; if "online" they are marked
synced immediately, if "offline" they sit in a sync queue until connectivity
returns. A same-equipment conflict between a locally-queued update and the
current server state is never silently overwritten -- it is flagged for
manager review (section 32).
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "oadrp.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS field_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    facility TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    occupancy_band TEXT,
    comfort_status TEXT,
    equipment_id TEXT,
    equipment_status TEXT,
    can_shift_load INTEGER,
    available_reduction_kw REAL,
    duration_min INTEGER,
    notes TEXT,
    sync_state TEXT NOT NULL DEFAULT 'SYNCED',   -- SYNCED | QUEUED | CONFLICT
    conflict_details TEXT
);

CREATE TABLE IF NOT EXISTS opt_outs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    preference TEXT NOT NULL,     -- PARTICIPATE | PARTICIPATE_ONLY_IF_NECESSARY | TEMPORARY_OPT_OUT
    reason TEXT,
    valid_from TEXT,
    valid_to TEXT
);

CREATE TABLE IF NOT EXISTS overrides (
    override_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    authorized_user TEXT NOT NULL,
    role TEXT NOT NULL,
    reason TEXT NOT NULL,
    affected_constraint TEXT NOT NULL,
    duration_min INTEGER NOT NULL,
    action TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approved_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    objective TEXT NOT NULL,
    predicted_peak_kw REAL,
    planned_peak_kw REAL,
    total_reduction_kw REAL,
    status TEXT NOT NULL,   -- APPROVED | MODIFIED | REJECTED
    approver TEXT,
    actions_json TEXT
);

CREATE TABLE IF NOT EXISTS equipment_server_state (
    equipment_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # seed equipment server state as AVAILABLE if empty
        from planner.facility_config import EQUIPMENT
        cur = conn.execute("SELECT COUNT(*) c FROM equipment_server_state")
        if cur.fetchone()["c"] == 0:
            now = datetime.now(timezone.utc).isoformat()
            conn.executemany(
                "INSERT INTO equipment_server_state (equipment_id, status, updated_at) VALUES (?,?,?)",
                [(e.equipment_id, "AVAILABLE", now) for e in EQUIPMENT],
            )


def save_field_observation(obs: dict, online: bool) -> dict:
    """
    Writes a field observation, applying the offline sync-queue logic.
    Returns the row as inserted, including sync_state and any conflict info.
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    sync_state = "SYNCED" if online else "QUEUED"
    conflict_details = None

    with get_conn() as conn:
        # Conflict check: does local equipment_status disagree with server state?
        if obs.get("equipment_id") and obs.get("equipment_status"):
            row = conn.execute(
                "SELECT status FROM equipment_server_state WHERE equipment_id=?",
                (obs["equipment_id"],),
            ).fetchone()
            if row is not None and row["status"] != obs["equipment_status"]:
                sync_state = "CONFLICT"
                conflict_details = (
                    f"Server state='{row['status']}' vs locally observed "
                    f"state='{obs['equipment_status']}'. Requires manager review before use."
                )

        cur = conn.execute(
            """INSERT INTO field_observations
               (created_at, facility, zone_id, occupancy_band, comfort_status,
                equipment_id, equipment_status, can_shift_load, available_reduction_kw,
                duration_min, notes, sync_state, conflict_details)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now, obs.get("facility", "Industrial Facility A"), obs["zone_id"],
             obs.get("occupancy_band"), obs.get("comfort_status"),
             obs.get("equipment_id"), obs.get("equipment_status"),
             int(bool(obs.get("can_shift_load"))), obs.get("available_reduction_kw"),
             obs.get("duration_min"), obs.get("notes"), sync_state, conflict_details),
        )
        obs_id = cur.lastrowid

    return {"id": obs_id, "sync_state": sync_state, "conflict_details": conflict_details}


def sync_queued_observations() -> dict:
    """Simulates connectivity returning: marks all QUEUED rows as SYNCED
    (CONFLICT rows are left for explicit manager resolution)."""
    init_db()
    with get_conn() as conn:
        cur = conn.execute("UPDATE field_observations SET sync_state='SYNCED' WHERE sync_state='QUEUED'")
        synced = cur.rowcount
        remaining_conflicts = conn.execute(
            "SELECT COUNT(*) c FROM field_observations WHERE sync_state='CONFLICT'"
        ).fetchone()["c"]
    return {"synced": synced, "remaining_conflicts": remaining_conflicts}


def resolve_conflict(observation_id: int, resolution: str, manager: str):
    """resolution: 'ACCEPT_LOCAL' or 'KEEP_SERVER'."""
    init_db()
    with get_conn() as conn:
        obs = conn.execute("SELECT * FROM field_observations WHERE id=?", (observation_id,)).fetchone()
        if obs is None:
            return {"ok": False, "error": "observation not found"}
        if resolution == "ACCEPT_LOCAL" and obs["equipment_id"]:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE equipment_server_state SET status=?, updated_at=? WHERE equipment_id=?",
                (obs["equipment_status"], now, obs["equipment_id"]),
            )
        conn.execute(
            "UPDATE field_observations SET sync_state='SYNCED', "
            "conflict_details = conflict_details || ' | Resolved by ' || ? || ': ' || ? WHERE id=?",
            (manager, resolution, observation_id),
        )
    return {"ok": True}


def set_equipment_status(equipment_id: str, status: str):
    init_db()
    with get_conn() as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO equipment_server_state (equipment_id, status, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(equipment_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
            (equipment_id, status, now),
        )


def get_equipment_status() -> dict:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT equipment_id, status FROM equipment_server_state").fetchall()
    return {r["equipment_id"]: r["status"] for r in rows}


def save_opt_out(zone_id: str, preference: str, reason: str, valid_from: str, valid_to: str):
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO opt_outs (created_at, zone_id, preference, reason, valid_from, valid_to) "
            "VALUES (?,?,?,?,?,?)",
            (now, zone_id, preference, reason, valid_from, valid_to),
        )


def get_active_opt_outs() -> list:
    """Returns zone_ids currently under TEMPORARY_OPT_OUT (most recent row per zone wins)."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT zone_id, preference FROM opt_outs o
               WHERE created_at = (SELECT MAX(created_at) FROM opt_outs o2 WHERE o2.zone_id = o.zone_id)"""
        ).fetchall()
    return [r["zone_id"] for r in rows if r["preference"] == "TEMPORARY_OPT_OUT"]


def save_override(authorized_user, role, reason, affected_constraint, duration_min, action):
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO overrides (timestamp, authorized_user, role, reason, affected_constraint, "
            "duration_min, action) VALUES (?,?,?,?,?,?,?)",
            (now, authorized_user, role, reason, affected_constraint, duration_min, action),
        )


def get_overrides() -> list:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM overrides ORDER BY timestamp DESC").fetchall()
    return [dict(r) for r in rows]


def save_approved_plan(objective, predicted_peak_kw, planned_peak_kw, total_reduction_kw, status, approver, actions_json):
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO approved_plans (created_at, objective, predicted_peak_kw, planned_peak_kw, "
            "total_reduction_kw, status, approver, actions_json) VALUES (?,?,?,?,?,?,?,?)",
            (now, objective, predicted_peak_kw, planned_peak_kw, total_reduction_kw, status, approver, actions_json),
        )


def get_field_observations(limit=50) -> list:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM field_observations ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
