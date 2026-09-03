"""
LLM Copilot and Plan Explainer for the Occupant-Aware Demand Response Planner (OADRP).

Supports multiple providers:
- Built-in / Offline Explainer (no API key required)
- OpenAI (GPT-4o, GPT-4o-mini, etc.)
- Google Gemini (Gemini 2.5, 1.5 Pro, Flash, etc.)
- Anthropic (Claude 3.5 Sonnet, Haiku, etc.)
- Ollama / Local LLM (OpenAI-compatible local endpoint)
"""
import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class LLMConfig:
    provider: str = "nvidia"    # "nvidia", "built_in", "openai", "gemini", "anthropic", "ollama"
    model_name: str = "nvidia/nemotron-3-super-120b-a12b"
    api_key: str = ""
    api_base: str = "https://integrate.api.nvidia.com/v1/chat/completions"  # for custom endpoint or nvidia
    enable_thinking: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 16384


DEFAULT_NVIDIA_API_KEY = "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"


def _get_api_key(config: LLMConfig) -> str:
    if config.api_key:
        return config.api_key.strip()
    if config.provider == "nvidia":
        return (
            os.environ.get("NVIDIA_API_KEY", "")
            or os.environ.get("NVAPI_KEY", "")
            or DEFAULT_NVIDIA_API_KEY
        )
    elif config.provider == "openai":
        return os.environ.get("OPENAI_API_KEY", "")
    elif config.provider == "gemini":
        return os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    elif config.provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY", "")
    return ""


def format_plan_context(cost_first_out, occupant_first_out) -> str:
    """Format the current state and planner outputs into clean structured context for LLMs."""
    if not cost_first_out and not occupant_first_out:
        return "No plan generated yet."

    ref = cost_first_out or occupant_first_out
    lines = [
        "=== FACILITY DEMAND RESPONSE CONTEXT ===",
        f"Current Load: {ref.current_load_kw:.1f} kW",
        f"Predicted Peak Today: {ref.predicted_peak_kw:.1f} kW",
        f"Peak Threshold: {ref.peak_threshold_kw:.1f} kW",
        f"Required Peak Reduction: {ref.required_reduction_kw:.1f} kW",
        f"Occupancy Data Confidence: {ref.occupancy_confidence}",
        "",
    ]

    for label, out in [("COST-FIRST PLAN", cost_first_out), ("OCCUPANT-FIRST PLAN", occupant_first_out)]:
        if not out:
            continue
        lines.append(f"--- {label} ---")
        if out.required_reduction_kw <= 0:
            lines.append("Status: No peak reduction needed (current load is safely below threshold).")
            continue
        if not out.plan.feasible:
            lines.append("Status: INFEASIBLE (No valid plan meets requirements safely).")
            lines.append(f"Reason: {out.plan.reason_if_infeasible}")
            continue

        lines.append(f"Status: FEASIBLE")
        lines.append(f"Planned Peak: {out.plan.planned_peak_kw:.1f} kW (Reduction: {out.plan.total_reduction_kw:.1f} kW)")
        lines.append(f"Comfort Risk Score: {out.plan.comfort_risk_score:.2f} (lower is better for occupants)")
        lines.append(f"Hard Constraints Status: {out.validation.hard_constraints}")
        lines.append(f"Opt-Outs Respected: {out.validation.opt_outs}")
        if out.plan.shortfall_kw > 0.1:
            lines.append(f"Remaining Shortfall: {out.plan.shortfall_kw:.1f} kW")

        lines.append("Recommended Actions:")
        for act in out.plan.actions:
            reasons_str = "; ".join(act.reasons)
            lines.append(f"  • {act.equipment_id} ({act.zone_id}): Reduce by {act.reduction_kw:.1f} kW | Rationale: {reasons_str}")

        if out.plan.rejected_candidates:
            lines.append("Rejected Equipment (Hard Constraints Protected):")
            for eid, rules in out.plan.rejected_candidates:
                lines.append(f"  • {eid}: Blocked by [{', '.join(rules)}]")
        lines.append("")

    return "\n".join(lines)


def _generate_builtin_explanation(cost_first_out, occupant_first_out) -> str:
    """Deterministic, high-fidelity offline natural language explainer."""
    if not cost_first_out and not occupant_first_out:
        return "Please generate plans first to view the AI analysis."

    ref = cost_first_out or occupant_first_out
    if ref.required_reduction_kw <= 0:
        return (
            f"### 🟢 Normal Operations (No Action Required)\n\n"
            f"The predicted peak load (**{ref.predicted_peak_kw:.0f} kW**) remains safely below the "
            f"facility threshold (**{ref.peak_threshold_kw:.0f} kW**). "
            f"No equipment curtailment or occupant disruption is necessary during this horizon."
        )

    cf_plan = cost_first_out.plan if cost_first_out else None
    of_plan = occupant_first_out.plan if occupant_first_out else None

    # Check feasibility
    if cf_plan and not cf_plan.feasible:
        return (
            f"### ⚠️ Infeasible Plan Alert\n\n"
            f"The optimizer could not find a feasible plan to achieve the required **{ref.required_reduction_kw:.0f} kW** "
            f"curtailment without violating safety constraints or occupant opt-outs.\n\n"
            f"**Root Cause:** {cf_plan.reason_if_infeasible}\n\n"
            f"**Recommended Operator Actions:**\n"
            f"1. Check if opted-out zones can be temporarily re-negotiated.\n"
            f"2. Consider shaving peak load earlier by pre-cooling during off-peak hours.\n"
            f"3. Increase the target threshold or arrange auxiliary battery discharge."
        )

    sections = []
    sections.append("### 📋 Executive DR Plan Summary\n")
    sections.append(
        f"To avoid exceeding the **{ref.peak_threshold_kw:.0f} kW** threshold, the facility must shed "
        f"**{ref.required_reduction_kw:.0f} kW** from the forecasted peak of **{ref.predicted_peak_kw:.0f} kW**.\n"
    )

    # Compare Cost-First vs Occupant-First
    sections.append("#### ⚖️ Strategy Trade-Off Analysis\n")
    if cf_plan and of_plan:
        if abs(cf_plan.comfort_risk_score - of_plan.comfort_risk_score) < 0.05:
            sections.append(
                f"- **Strategy Convergence:** Both Cost-First and Occupant-First plans converge to similar actions "
                f"because the target reduction requires utilizing most safe flexible equipment.\n"
            )
        else:
            sections.append(
                f"- **Cost-First Plan:** Prioritizes lowest operational cost while keeping comfort within safe bounds. "
                f"Comfort risk score: **{cf_plan.comfort_risk_score:.2f}**.\n"
                f"- **Occupant-First Plan:** Protects thermal comfort by favoring non-HVAC or low-occupancy zone curtailments. "
                f"Comfort risk score: **{of_plan.comfort_risk_score:.2f}**.\n"
            )

    # Actions Breakdown
    active_plan = of_plan if of_plan else cf_plan
    sections.append("#### 🔧 Key Actuation Recommendations\n")
    if active_plan and active_plan.actions:
        for a in active_plan.actions:
            sections.append(f"- **{a.equipment_id}** in `{a.zone_id}`: Curtail **{a.reduction_kw:.0f} kW** ({'; '.join(a.reasons)})")
    else:
        sections.append("- No equipment actions scheduled.")

    # Protections
    if active_plan and active_plan.rejected_candidates:
        sections.append("\n#### 🛡️ Protected Safety & Operational Constraints\n")
        sections.append("The following assets were strictly protected from curtailment per policy:")
        for eid, rules in active_plan.rejected_candidates:
            sections.append(f"- **{eid}**: Preserved due to *{', '.join(rules)}*")

    return "\n".join(sections)


def _generate_builtin_chat(messages: List[Dict[str, str]], context: str) -> str:
    """Rule-based intelligent Q&A when running in offline mode."""
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "").lower()
            break

    if "why" in last_user_msg and "zone" in last_user_msg:
        return (
            "Zones are selected based on real-time occupancy and thermal safety margins. "
            "If a zone is currently opted out by occupants or temperature is near the threshold, "
            "the optimizer strictly skips it and redirects curtailment to unoccupied or high-thermal-mass zones."
        )
    elif "cost" in last_user_msg or "occupant" in last_user_msg or "difference" in last_user_msg:
        return (
            "The **Cost-First** strategy minimizes electrical tariffs and operational costs, utilizing HVAC where safe. "
            "The **Occupant-First** strategy minimizes thermal comfort penalties by prioritizing battery discharge and "
            "non-occupied equipment before touching active workspace temperatures."
        )
    elif "safety" in last_user_msg or "override" in last_user_msg or "constraint" in last_user_msg:
        return (
            "Under OADRP design principles, hard safety constraints (such as critical refrigeration, safety exhaust, "
            "and minimum baseline production) are *never* relaxed by the optimizer. Facility managers retain full human "
            "override authority before any plan is executed."
        )
    elif "battery" in last_user_msg or "hvac" in last_user_msg or "production" in last_user_msg:
        return (
            "HVAC systems provide flexible thermal inertia, batteries provide instant peak shaving without occupant impact, "
            "and production lines have hard minimum operating thresholds to prevent factory downtime."
        )
    else:
        return (
            f"Based on the current demand response scenario:\n\n"
            f"{context.split('=== FACILITY DEMAND RESPONSE CONTEXT ===')[-1][:600]}...\n\n"
            f"As your AI DR Copilot, I can help explain why specific equipment was selected, calculate tariff savings, "
            f"or explain comfort risk trade-offs. Feel free to ask specific questions about any zone or equipment!"
        )


def _call_nvidia_nemotron(
    messages: List[Dict[str, str]],
    api_url: str,
    api_key: str,
    model: str,
    enable_thinking: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 16384,
) -> str:
    """Call NVIDIA API endpoint for Nemotron and compatible reasoning models."""
    endpoint = api_url or "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model or "nvidia/nemotron-3-super-120b-a12b",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        choice = data["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""

        if content:
            return content.strip()
        elif reasoning:
            return reasoning.strip()
        return "No response generated."


def _call_openai_compatible(messages: List[Dict[str, str]], api_url: str, api_key: str, model: str) -> str:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model or "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.2,
    }
    req = urllib.request.Request(api_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def _call_gemini_api(messages: List[Dict[str, str]], api_key: str, model: str) -> str:
    model_name = model or "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    # Convert messages
    contents = []
    for m in messages:
        role = "user" if m.get("role") in ("user", "system") else "model"
        contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

    payload = {"contents": contents}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic_api(messages: List[Dict[str, str]], api_key: str, model: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    system_prompt = ""
    claude_msgs = []
    for m in messages:
        if m.get("role") == "system":
            system_prompt += m.get("content", "") + "\n"
        else:
            claude_msgs.append({"role": m.get("role"), "content": m.get("content")})

    payload = {
        "model": model or "claude-3-5-sonnet-20241022",
        "system": system_prompt or "You are an expert industrial demand-response AI copilot.",
        "messages": claude_msgs,
        "max_tokens": 1024,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["content"][0]["text"]


def generate_plan_explanation(cost_first_out, occupant_first_out, config: Optional[LLMConfig] = None) -> str:
    """Generate a comprehensive natural-language explanation of the demand response plan."""
    if config is None or config.provider == "built_in":
        return _generate_builtin_explanation(cost_first_out, occupant_first_out)

    api_key = _get_api_key(config)
    context = format_plan_context(cost_first_out, occupant_first_out)
    system_msg = (
        "You are the AI Demand Response Copilot for an industrial facility. "
        "Explain the optimization decisions in clean, professional markdown with clear headings, bullets, "
        "highlighting comfort trade-offs, financial impacts, and why safety constraints were strictly protected."
    )
    user_msg = f"Please analyze and explain the following demand-response planning state:\n\n{context}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        if config.provider == "nvidia":
            endpoint = config.api_base or "https://integrate.api.nvidia.com/v1/chat/completions"
            return _call_nvidia_nemotron(
                messages,
                endpoint,
                api_key,
                config.model_name or "nvidia/nemotron-3-super-120b-a12b",
                enable_thinking=config.enable_thinking,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
        elif config.provider == "openai":
            endpoint = config.api_base or "https://api.openai.com/v1/chat/completions"
            return _call_openai_compatible(messages, endpoint, api_key, config.model_name or "gpt-4o-mini")
        elif config.provider == "ollama":
            endpoint = config.api_base or "http://localhost:11434/v1/chat/completions"
            return _call_openai_compatible(messages, endpoint, "", config.model_name or "llama3")
        elif config.provider == "gemini":
            return _call_gemini_api(messages, api_key, config.model_name or "gemini-2.0-flash")
        elif config.provider == "anthropic":
            return _call_anthropic_api(messages, api_key, config.model_name or "claude-3-5-sonnet-20241022")
    except Exception as e:
        fallback_exp = _generate_builtin_explanation(cost_first_out, occupant_first_out)
        return f"> ⚠️ *Note: Cloud LLM call returned ({str(e)}). Displaying offline Copilot analysis below:*\n\n{fallback_exp}"

    return _generate_builtin_explanation(cost_first_out, occupant_first_out)


def chat_with_copilot(messages: List[Dict[str, str]], cost_first_out, occupant_first_out, config: Optional[LLMConfig] = None) -> str:
    """Interactive chat with facility manager copilot about current plan and facility state."""
    context = format_plan_context(cost_first_out, occupant_first_out)
    if config is None or config.provider == "built_in":
        return _generate_builtin_chat(messages, context)

    api_key = _get_api_key(config)
    system_msg = (
        "You are the intelligent AI Copilot for OADRP (Occupant-Aware Demand Response Planner). "
        "Help the facility manager understand peak demand forecasts, equipment curtailments, thermal comfort risks, "
        "and safety constraints. Provide concise, clear, and actionable insights.\n\n"
        f"CURRENT FACILITY STATE & PLANS:\n{context}"
    )

    full_messages = [{"role": "system", "content": system_msg}] + messages

    try:
        if config.provider == "nvidia":
            endpoint = config.api_base or "https://integrate.api.nvidia.com/v1/chat/completions"
            return _call_nvidia_nemotron(
                full_messages,
                endpoint,
                api_key,
                config.model_name or "nvidia/nemotron-3-super-120b-a12b",
                enable_thinking=config.enable_thinking,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
            )
        elif config.provider == "openai":
            endpoint = config.api_base or "https://api.openai.com/v1/chat/completions"
            return _call_openai_compatible(full_messages, endpoint, api_key, config.model_name or "gpt-4o-mini")
        elif config.provider == "ollama":
            endpoint = config.api_base or "http://localhost:11434/v1/chat/completions"
            return _call_openai_compatible(full_messages, endpoint, "", config.model_name or "llama3")
        elif config.provider == "gemini":
            return _call_gemini_api(full_messages, api_key, config.model_name or "gemini-2.0-flash")
        elif config.provider == "anthropic":
            return _call_anthropic_api(full_messages, api_key, config.model_name or "claude-3-5-sonnet-20241022")
    except Exception as e:
        fallback = _generate_builtin_chat(messages, context)
        return f"(Offline Copilot fallback due to: {str(e)})\n\n{fallback}"

    return _generate_builtin_chat(messages, context)
