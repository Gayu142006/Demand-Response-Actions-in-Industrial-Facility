# Scope and Limitations

This project was built end-to-end in a single session against a 97-section
specification. To deliver something *real and fully working* rather than a
shallow pass over everything, scope was deliberately narrowed. This
document is the honest record of what was cut and why, per the intent of
the spec's own emphasis on multi-source validation and honest reporting.

## Built and verified working

- Synthetic data generator producing 2,880 fifteen-minute observations
  across 30 days, 10 named scenario types, reproducible via a fixed seed
  (`scripts/generate_data.py`).
- Full constraint engine with 6 hard-constraint rules, independently
  re-validated post-optimization (`planner/constraints.py`,
  `planner/validator.py`).
- A genuine OR-Tools CP-SAT optimizer (not a greedy heuristic or stub) with
  two real objective weightings (`planner/optimizer.py`).
- Two forecasting approaches (historical-average and RandomForest) with
  honestly measured MAE/RMSE/peak-error, including a case where the simpler
  historical-average model's peak error is *lower* than the ML model's
  overall MAE would suggest -- reported as-is, not massaged.
- A 5-page Streamlit application (dashboard, field capture with real
  offline/online queue + conflict detection, planner with side-by-side
  objective comparison, approvals with audit-logged overrides, evaluation
  results viewer).
- A SQLite persistence layer with a real offline-sync conflict-detection
  mechanism (not a hand-waved "TODO: sync logic").
- 34 passing pytest tests covering the constraint engine, optimizer,
  planner engine (including all named failure cases), baseline/forecast
  math, and the database/offline-sync layer.
- A full 30-day baseline-vs-cost-first-vs-occupant-first evaluation
  experiment producing real numbers in
  `experiments/results/evaluation_summary.json` and
  `daily_comparison.csv`, including a finding (objective convergence under
  extreme scenarios) that was not anticipated going in.

## Explicitly cut or simplified (not silently skipped)

- **No FastAPI/REST backend.** The spec sketches an API layer; this
  prototype calls `planner_engine.run_planner()` directly from the
  Streamlit app and the evaluation script. Adding a thin FastAPI wrapper
  around the same `planner_engine` functions would be a small, low-risk
  follow-up -- the business logic is already decoupled from the UI.
- **No live IoT/BMS/meter integration.** All data is synthetic
  (`scripts/generate_data.py`). No real facility telemetry exists for this
  project.
- **No native mobile app.** The field-capture workflow (including the
  offline queue) is implemented in the Streamlit web app via an "online/
  offline" toggle that simulates connectivity, not a real mobile client
  with real network-loss handling.
- **Opt-out preference is binary in practice.** The spec describes three
  tiers (`PARTICIPATE`, `PARTICIPATE_ONLY_IF_NECESSARY`,
  `TEMPORARY_OPT_OUT`); the optimizer today treats the first two the same
  (available) and the third as a hard exclusion. Making
  `PARTICIPATE_ONLY_IF_NECESSARY` a genuine soft constraint (usable only
  when no other feasible plan exists) is a one-function change to
  `optimizer.py` but wasn't implemented to keep the constraint model easy
  to audit within this session.
- **No authentication/RBAC.** See `docs/privacy.md`.
- **No multi-facility / multi-tenant model.** One synthetic facility, six
  zones, nine equipment items -- deliberately small enough to reason about
  and test exhaustively, not a claim that the architecture doesn't
  generalize.
- **No Jupyter notebooks.** Exploratory analysis was done directly via
  `experiments/evaluation.py`, which is a plain script rather than a
  notebook, so it can be run headlessly and is covered by the same
  reproducibility guarantees as the rest of the code.
- **No PPTX/slide deliverable.** This markdown documentation plus the live
  Evaluation page in the app serve the same purpose for this prototype.
- **No Docker packaging.** `requirements.txt` plus the documented `pip
  install` + `streamlit run` steps are sufficient for local reproduction;
  containerization was deprioritized in favor of finishing the planning
  logic and tests correctly.

## Known rough edges

- The evaluation experiment's "sustained DR event window" (the block of
  intervals treated as needing curtailment) is defined as the contiguous
  run of intervals above the peak threshold on a given day. This is a
  reasonable modeling choice but is a simplification of how a real facility
  would decide when to start/stop an event.
- The temperature-to-load relationship in `planner/constraints.py`
  (`projected_rise = reduction_kw * 0.06`) is a simple linear
  approximation calibrated to produce plausible behavior across the test
  scenarios, not a validated thermal model.
