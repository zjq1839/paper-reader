import os

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import get_library_structure, read_section_content, report_status, update_workspace


def build_agent():
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY 未设置")

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    model = ChatNVIDIA(
        model=os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b"),
        api_key=api_key,
        temperature=float(os.environ.get("NVIDIA_TEMPERATURE", "0.6")),
        top_p=float(os.environ.get("NVIDIA_TOP_P", "0.95")),
        max_completion_tokens=int(os.environ.get("NVIDIA_MAX_TOKENS", "204800")),
    )

    tools = [report_status, get_library_structure, read_section_content, update_workspace]
    return create_agent(model=model, tools=tools, system_prompt=SystemMessage(content=SYSTEM_PROMPT))


agent = build_agent()
