import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.database import init_db

st.set_page_config(page_title="Occupant-Aware Demand Response Planner", layout="wide")
init_db()

st.sidebar.title("⚡ OADRP")
st.sidebar.caption("Occupant-Aware Demand Response Planner")
page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Field Capture",
        "Planner",
        "Approvals",
        "Evaluation",
        "IoT Gateway",
        "Thermal Lab",
        "Validation & Feedback",
        "Equipment Fatigue",
    ],
    label_visibility="collapsed",
)

if "online" not in st.session_state:
    st.session_state.online = True
st.sidebar.divider()
st.session_state.online = st.sidebar.toggle("Connectivity (online)", value=st.session_state.online)
if not st.session_state.online:
    st.sidebar.warning("Offline mode active — field data is queued locally.")

st.sidebar.divider()
st.sidebar.markdown("### 🤖 AI Copilot Settings")
llm_provider_label = st.sidebar.selectbox(
    "LLM Provider",
    [
        "NVIDIA Nemotron (Super 120B)",
        "Built-in (Offline Explainer)",
        "OpenAI",
        "Google Gemini",
        "Anthropic",
        "Ollama / Local LLM",
    ],
    index=0,
)

provider_map = {
    "NVIDIA Nemotron (Super 120B)": "nvidia",
    "Built-in (Offline Explainer)": "built_in",
    "OpenAI": "openai",
    "Google Gemini": "gemini",
    "Anthropic": "anthropic",
    "Ollama / Local LLM": "ollama",
}

chosen_provider = provider_map[llm_provider_label]
api_key = ""
api_base = ""
model_name = ""
enable_thinking = True

if chosen_provider == "nvidia":
    model_name = st.sidebar.text_input("NVIDIA Model", value="nvidia/nemotron-3-super-120b-a12b")
    api_key = st.sidebar.text_input(
        "NVIDIA API Key",
        type="password",
        value="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",
        help="NVIDIA API Key for Nemotron endpoints",
    )
    api_base = st.sidebar.text_input(
        "NVIDIA Base URL",
        value="https://integrate.api.nvidia.com/v1/chat/completions",
    )
    enable_thinking = st.sidebar.toggle("Enable Thinking (Reasoning)", value=True)
elif chosen_provider == "built_in":
    st.sidebar.caption("⚡ *Zero-config offline mode with built-in DR knowledge & reasoning engine.*")
elif chosen_provider == "openai":
    model_name = st.sidebar.text_input("OpenAI Model", value="gpt-4o-mini")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")
elif chosen_provider == "gemini":
    model_name = st.sidebar.text_input("Gemini Model", value="gemini-2.0-flash")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
elif chosen_provider == "anthropic":
    model_name = st.sidebar.text_input("Anthropic Model", value="claude-3-5-sonnet-20241022")
    api_key = st.sidebar.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
elif chosen_provider == "ollama":
    api_base = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434/v1/chat/completions")
    model_name = st.sidebar.text_input("Local Model Name", value="llama3")

from planner.llm_assistant import LLMConfig
st.session_state["llm_config"] = LLMConfig(
    provider=chosen_provider,
    model_name=model_name,
    api_key=api_key,
    api_base=api_base,
    enable_thinking=enable_thinking,
)

if page == "Dashboard":
    from app.views import dashboard
    dashboard.render()
elif page == "Field Capture":
    from app.views import field_capture
    field_capture.render()
elif page == "Planner":
    from app.views import planner as planner_page
    planner_page.render()
elif page == "Approvals":
    from app.views import approvals
    approvals.render()
elif page == "Evaluation":
    from app.views import evaluation
    evaluation.render()
elif page == "IoT Gateway":
    from app.views import iot_gateway_view
    iot_gateway_view.render()
elif page == "Thermal Lab":
    from app.views import thermal_lab_view
    thermal_lab_view.render()
elif page == "Validation & Feedback":
    from app.views import validation_view
    validation_view.render()
elif page == "Equipment Fatigue":
    from app.views import equipment_fatigue_view
    equipment_fatigue_view.render()
