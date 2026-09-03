# Architecture

## Overview

The Occupant-Aware Demand Response Planner (OADRP) is a decision-support
prototype: it recommends demand-response actions to a human facility
manager, who approves, modifies, rejects, or overrides them. It never
directly actuates equipment. This is a deliberate scope decision (see
`docs/scope_and_limitations.md`).

```
                    ┌─────────────────────┐
 synthetic world -> │  scripts/generate_   │ -> data/generated/*.csv
 model (no real     │  data.py             │    (load, occupancy, equipment,
 telemetry exists)  └─────────────────────┘     opt-outs, day scenarios)
                              │
                              ▼
                    ┌─────────────────────┐
                    │  planner/            │
                    │  - facility_config   │  static zone/equipment/tariff model
                    │  - constraints        │  hard-constraint validator (never relaxed)
                    │  - optimizer          │  OR-Tools CP-SAT, two objectives
                    │  - baseline           │  no-action counterfactual + tariff math
                    │  - demand_forecast    │  historical-average + RandomForest
                    │  - validator          │  final independent safety pass
                    │  - planner_engine     │  orchestrates the above (single entrypoint)
                    └─────────────────────┘
                        │                │
                        ▼                ▼
              ┌──────────────┐   ┌──────────────────┐
              │ app/ (Streamlit)│ │ experiments/       │
              │ - dashboard     │ │  evaluation.py     │
              │ - field_capture │ │  runs planner_engine│
              │ - planner       │ │  over the full     │
              │ - approvals     │ │  synthetic dataset  │
              │ - evaluation    │ │  and writes         │
              │                 │ │  results/*.csv+json │
              │ app/database.py │ └──────────────────┘
              │  SQLite: field
              │  observations,
              │  opt-outs,
              │  overrides,
              │  approved plans
              └──────────────┘
```

## Why this shape

- **`planner_engine.run_planner()` is the single source of truth.** Both the
  interactive app and the offline evaluation experiment call the exact same
  function, so the numbers in `experiments/results/` are guaranteed to be
  what the app would actually produce for the same inputs -- not a separate,
  hand-tuned "demo path".

- **Hard constraints are enforced in `constraints.py` and re-checked in
  `validator.py`.** The optimizer only ever sees candidates that already
  passed the hard-constraint filter; it cannot "solve around" a safety rule
  by assigning it a large penalty, because rejected equipment is removed
  from the decision variables entirely, not penalized. `validator.py` then
  re-validates the optimizer's own output independently, as a second,
  cheaper check before the plan reaches the approval UI.

- **SQLite, not a mocked backend.** `app/database.py` is a real, if small,
  persistence layer with an actual offline/online distinction: observations
  saved while `online=False` are marked `QUEUED`; a same-equipment
  disagreement between a queued observation and the current server state is
  marked `CONFLICT` and is never silently auto-resolved (see
  `docs/failure_modes.md`, case 6).

- **OR-Tools CP-SAT, not a greedy heuristic.** The optimizer is a genuine
  integer program: decision variables are per-equipment reduction amounts
  (scaled to hundredths of a kW for solver precision), the objective
  combines a shortfall penalty with weighted comfort/disruption/preference
  penalties, and infeasibility is reported explicitly via a `shortfall`
  variable rather than forced to zero.

## Data flow for one planning cycle

1. `app/state.py` (or `experiments/evaluation.py`) picks a snapshot: current
   load, zone temperatures, occupancy band, and live equipment/opt-out state
   from `app/database.py`.
2. `planner_engine.run_planner()` computes `required_reduction_kw =
   max(0, predicted_peak - threshold)`.
3. If `required_reduction_kw <= 0`, an empty (feasible) plan is returned
   immediately -- no equipment is touched.
4. Otherwise `optimizer.optimize()` filters equipment through
   `constraints.validate_hard_constraints()`, computes a safe per-equipment
   ceiling (capped further by temperature headroom for HVAC), and solves the
   CP-SAT model for the objective in play.
5. `validator.validate_plan()` re-checks the result.
6. The app renders the plan with per-action reasons; the manager approves,
   modifies, rejects, or authorizes an override (all logged to SQLite).

## What is *not* built

See `docs/scope_and_limitations.md` for the explicit, honest list of what
was cut to fit this as a single-session prototype (e.g. no live IoT/BMS
integration, no mobile app, no multi-facility multi-tenant auth model, no
production deployment hardening).
