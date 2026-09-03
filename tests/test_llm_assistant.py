# pyrefly: ignore [missing-import]
import pytest
from planner.llm_assistant import (
    LLMConfig,
    format_plan_context,
    generate_plan_explanation,
    chat_with_copilot,
)
from planner.planner_engine import run_planner


@pytest.fixture
def sample_plans():
    cost_first = run_planner(
        current_load_kw=680.0,
        predicted_peak_kw=820.0,
        zone_temperatures={"Zone_1": 22.5, "Zone_2": 23.0, "Zone_3": 21.0},
        equipment_status={"HVAC_1": "AVAILABLE", "BATTERY_1": "AVAILABLE"},
        opted_out_zones=set(),
        current_production_load_kw=100.0,
        objective="COST_FIRST",
        peak_threshold_kw=750.0,
    )
    occupant_first = run_planner(
        current_load_kw=680.0,
        predicted_peak_kw=820.0,
        zone_temperatures={"Zone_1": 22.5, "Zone_2": 23.0, "Zone_3": 21.0},
        equipment_status={"HVAC_1": "AVAILABLE", "BATTERY_1": "AVAILABLE"},
        opted_out_zones=set(),
        current_production_load_kw=100.0,
        objective="OCCUPANT_FIRST",
        peak_threshold_kw=750.0,
    )
    return cost_first, occupant_first


def test_format_plan_context(sample_plans):
    cf, of = sample_plans
    context = format_plan_context(cf, of)
    assert "FACILITY DEMAND RESPONSE CONTEXT" in context
    assert "COST-FIRST PLAN" in context
    assert "OCCUPANT-FIRST PLAN" in context
    assert "Predicted Peak Today" in context


def test_builtin_explanation_generation(sample_plans):
    cf, of = sample_plans
    explanation = generate_plan_explanation(cf, of, LLMConfig(provider="built_in"))
    assert "Executive DR Plan Summary" in explanation
    assert "Trade-Off Analysis" in explanation
    assert len(explanation) > 100


def test_builtin_chat_responses(sample_plans):
    cf, of = sample_plans
    res1 = chat_with_copilot([{"role": "user", "content": "Why was Zone 1 selected?"}], cf, of)
    assert "occupancy" in res1.lower() or "thermal" in res1.lower()

    res2 = chat_with_copilot([{"role": "user", "content": "What is the difference between cost and occupant plans?"}], cf, of)
    assert "cost-first" in res2.lower() or "occupant-first" in res2.lower()


def test_cloud_provider_fallback_on_network_or_missing_key(sample_plans):
    cf, of = sample_plans
    # Invalid config should safely fall back with a descriptive notice
    cfg = LLMConfig(provider="openai", model_name="gpt-4o-mini", api_key="invalid-key-for-testing", api_base="http://127.0.0.1:9999/v1")
    explanation = generate_plan_explanation(cf, of, cfg)
    assert "Offline Copilot" in explanation or "Executive DR Plan Summary" in explanation


def test_nvidia_nemotron_config_defaults():
    cfg = LLMConfig()
    assert cfg.provider == "nvidia"
    assert cfg.model_name == "nvidia/nemotron-3-super-120b-a12b"
    assert "integrate.api.nvidia.com" in cfg.api_base
    assert cfg.enable_thinking is True


def test_nvidia_nemotron_fallback_on_error(sample_plans):
    cf, of = sample_plans
    cfg = LLMConfig(
        provider="nvidia",
        model_name="nvidia/nemotron-3-super-120b-a12b",
        api_key="invalid-test-key",
        api_base="http://127.0.0.1:9999/v1/chat/completions",
    )
    explanation = generate_plan_explanation(cf, of, cfg)
    assert "Executive DR Plan Summary" in explanation or "Note: Cloud LLM call returned" in explanation

    chat_res = chat_with_copilot([{"role": "user", "content": "Explain safety rules"}], cf, of, cfg)
    assert "safety" in chat_res.lower() or "offline" in chat_res.lower()

