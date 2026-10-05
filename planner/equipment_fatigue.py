"""
Equipment Cycling & Fatigue Tracking Engine.

In industrial demand response, repeatedly cycling large equipment (compressors,
pumps, chillers, and battery charge cycles) causes mechanical wear, contactor
arcing, and thermal stress on windings.

This module tracks:
  - Total curtailment activations per piece of equipment
  - Cumulative curtailed kilowatt-hours
  - Cycling frequency (starts/stops within rolling windows)
  - Fatigue index (0.0 to 100.0)
  - Dynamic wear penalty to incentivize rotational fairness among redundant equipment
    (e.g., COMPRESSOR_A vs COMPRESSOR_B)
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

FATIGUE_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "equipment_fatigue.json"
FATIGUE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class EquipmentWearRecord:
    equipment_id: str
    equipment_type: str
    total_events: int = 0
    total_curtailed_kwh: float = 0.0
    last_event_timestamp: Optional[str] = None
    fatigue_index: float = 0.0          # 0.0 (fresh) to 100.0 (needs maintenance)
    max_recommended_cycles_per_month: int = 30
    cycles_this_month: int = 0
    wear_penalty_multiplier: float = 1.0  # multiplier to add to disruption penalty


class EquipmentFatigueTracker:
    """Tracks and calculates wear penalties to balance equipment utilization."""

    def __init__(self, persistence_file: Optional[Path] = None):
        self.file_path = persistence_file or FATIGUE_DATA_FILE
        self.records: Dict[str, EquipmentWearRecord] = {}
        self._load()

    def _load(self):
        if self.file_path.exists():
            try:
                with open(self.file_path) as f:
                    data = json.load(f)
                    for eid, rec in data.items():
                        self.records[eid] = EquipmentWearRecord(**rec)
            except Exception:
                self.records = {}

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump({eid: asdict(r) for eid, r in self.records.items()}, f, indent=2)

    def get_or_create(self, equipment_id: str, equipment_type: str) -> EquipmentWearRecord:
        if equipment_id not in self.records:
            max_cycles = 40 if equipment_type in ("PUMP", "LIGHTING") else 20
            self.records[equipment_id] = EquipmentWearRecord(
                equipment_id=equipment_id,
                equipment_type=equipment_type,
                max_recommended_cycles_per_month=max_cycles,
            )
        return self.records[equipment_id]

    def record_curtailment(self, equipment_id: str, equipment_type: str, reduction_kw: float, duration_hours: float = 0.5):
        """Record an equipment curtailment action and update fatigue metrics."""
        rec = self.get_or_create(equipment_id, equipment_type)
        rec.total_events += 1
        rec.cycles_this_month += 1
        curtailed_kwh = reduction_kw * duration_hours
        rec.total_curtailed_kwh += curtailed_kwh
        rec.last_event_timestamp = datetime.now(timezone.utc).isoformat()

        # Update fatigue index: based on cycles relative to monthly limit
        cycle_ratio = rec.cycles_this_month / max(1, rec.max_recommended_cycles_per_month)
        rec.fatigue_index = min(100.0, round(cycle_ratio * 70.0 + (rec.total_events * 0.5), 1))

        # Dynamic wear penalty: when fatigue > 50%, add penalty to encourage rotating to peers
        if rec.fatigue_index > 50.0:
            rec.wear_penalty_multiplier = round(1.0 + (rec.fatigue_index - 50.0) / 25.0, 2)
        else:
            rec.wear_penalty_multiplier = 1.0

        self.save()

    def get_wear_penalty(self, equipment_id: str) -> float:
        """Returns extra disruption penalty (0..50) for heavily fatigued equipment."""
        rec = self.records.get(equipment_id)
        if not rec:
            return 0.0
        if rec.fatigue_index > 60.0:
            return (rec.fatigue_index - 60.0) * 0.75
        return 0.0

    def get_fleet_status(self) -> List[Dict[str, Any]]:
        """Return wear summary table for all tracked equipment."""
        return [asdict(r) for r in self.records.values()]
