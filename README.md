# Occupant-Aware Demand Response Planner (OADRP)

A decision-support prototype that recommends demand-response actions
(reducing electrical peak demand) for an industrial facility while
protecting occupant comfort and hard safety constraints. A human facility
manager always approves, modifies, rejects, or overrides the plan — the
system never directly actuates equipment.

Built from a 97-section specification in a single session. See
**`docs/scope_and_limitations.md`** for an honest account of what's fully
built vs. simplified, and **`docs/failure_modes.md`** for how each named
failure case is handled and tested.

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset (deterministic, seeded)
python scripts/generate_data.py

# 2. Run the test suite
pytest tests/ -v

# 3. Run the baseline-vs-planner evaluation experiment
python experiments/evaluation.py

# 4. Launch the app
streamlit run app/main.py
```

## What's in here

| Path | What |
|---|---|
| `planner/` | Core logic: facility model, constraints, OR-Tools optimizer, forecasting, baseline, validator, top-level engine, LLM assistant |
| `planner/thermal_model.py` | 1R1C and 2R2C lumped-capacitance thermal physics model replacing linear approximations with wall-capacitance buffering |
| `planner/iot_gateway.py` | BMS edge-gateway telemetry ingestion layer with BACnet/IP, Modbus TCP/RTU, and MQTT protocol adapters |
| `planner/equipment_fatigue.py` | Mechanical cycling fatigue and wear-tracking engine to rotate curtailments and prevent motor/chiller degradation |
| `planner/llm_assistant.py` | Multi-provider AI Copilot (NVIDIA Nemotron, Built-in offline explainer, OpenAI, Gemini, Claude, Ollama) |
| `scripts/generate_data.py` | Synthetic 30-day / 2,880-interval dataset generator with 10 named scenario types |
| `app/` | 9-page Streamlit application (Dashboard, Field Capture, Planner + AI Copilot, Approvals, Evaluation, IoT Gateway, Thermal Lab, Validation, Equipment Fatigue) |
| `app/api.py` | Starlette/ASGI REST API backend exposing endpoints for planning, telemetry ingestion, thermal simulation, and validation |
| `experiments/evaluation.py` | Full empirical evaluation: cost-first vs occupant-first divergence on moderate events & per-day forecast peak errors |
| `experiments/evaluation_notebook.ipynb` | Executable Jupyter Notebook companion deliverable with interactive Plotly & Matplotlib figures |
| `tests/` | 59 unit & integration tests across all modules (constraints, optimizer, thermal models, IoT gateway, fatigue, validation) |
| `validation/protocol.py` | Programmatic multi-stakeholder validation questionnaire & scoring rubric (Usability, Trust, Safety, Explainability, Privacy) |
| `validation/adaptive_tuner.py` | Feedback-driven adaptive calibration that automatically tunes optimizer weights and safety margins from stakeholder scores |
| `docs/` | Architecture, failure modes, privacy, honest scope/limitations |

## AI Copilot & Decision Explainer

OADRP includes an AI Copilot in the **Planner** view to help facility managers interpret complex demand response trade-offs:
- **Executive Plan Analysis**: Explains curtailment decisions, comfort risk scores, tariff savings, and protected safety constraints in clear natural language.
- **Interactive Q&A Chat**: Facility managers can ask operational questions (e.g., *"Why was Zone 1 chosen over Zone 2?"*, *"What happens if we lose Battery Unit 1?"*).
- **Multi-Provider Support**: Switch seamlessly in the sidebar between **NVIDIA Nemotron (`nvidia/nemotron-3-super-120b-a12b`)**, the **Built-in Offline Explainer** (zero API keys needed), **OpenAI**, **Google Gemini**, **Anthropic Claude**, or local **Ollama** models. Supported with deep reasoning / thinking tokens (`chat_template_kwargs: {"enable_thinking": True}`).

## Headline result (from `experiments/results/evaluation_summary.json`)

On the 4 of 30 synthetic days that actually crossed the 750 kW peak
threshold, the planner achieved a **14.3% average peak reduction** with
**zero hard-constraint or opt-out violations** and **zero comfort-violation
minutes**, vs. a no-action baseline. Across *all* 30 days (most of which
never approach the threshold at all — a realistic mix, not cherry-picked),
the average peak reduction is much smaller, which is reported honestly
rather than only showing the flattering event-day number. Cost-first and
occupant-first plans converge under extreme scenarios (all available
flexibility is needed regardless of objective) and diverge under moderate
events — see `docs/failure_modes.md` for the detail.

## Design principle carried through every layer

Hard safety/operational constraints (critical equipment, opted-out zones,
equipment under maintenance, minimum production load, absolute temperature
safety limits) are **never relaxed by objective weighting**. The optimizer
only ever chooses *how much* of an already-safe action to take; a candidate
that fails a hard constraint is removed from the decision variables
entirely, never penalized into submission. This is enforced in code
(`planner/constraints.py`), re-checked independently after optimization
(`planner/validator.py`), and tested explicitly
(`tests/test_planner_engine.py::test_objective_never_overrides_hard_constraints`).
