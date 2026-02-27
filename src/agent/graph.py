import os
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from src.agent.tools import (
    get_library_structure,
    hybrid_search,
    read_section_content,
    update_workspace,
    report_status,
    search_episodic_memory,
    update_semantic_knowledge
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    working_workspace: str
    semantic_directives: str
    episodic_events: list[str]
    summary: str

CORE_FILE = os.path.join(os.getcwd(), "docs", "DEEPSCHOLAR_CORE.md")

def estimate_tokens(messages: list[BaseMessage]) -> int:
    total_chars = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total_chars += len(msg.content)
        elif isinstance(msg.content, list):
             for part in msg.content:
                 if isinstance(part, str):
                     total_chars += len(part)
                 elif isinstance(part, dict) and "text" in part:
                     total_chars += len(part["text"])
    return total_chars // 4

MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "32000"))
COMPACTION_THRESHOLD = int(MAX_CONTEXT_TOKENS * 0.95)

def load_core_constraints():
    if os.path.exists(CORE_FILE):
        with open(CORE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return """
# DEEPSCHOLAR CORE PROTOCOLS
1. Always prioritize evidence from the provided literature.
2. Use the hybrid retrieval system to find information.
3. Maintain a structured workspace in 'working_workspace'.
4. If conflicts arise between papers, use the Conflict Resolution Protocol.
"""

tools = [
    get_library_structure,
    hybrid_search,
    read_section_content,
    update_workspace,
    report_status,
    search_episodic_memory,
    update_semantic_knowledge
]

nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "").strip().strip('"').strip("'")
if not nvidia_api_key:
    raise RuntimeError("NVIDIA_API_KEY 未设置")

llm = ChatNVIDIA(
    model=os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b"),
    api_key=nvidia_api_key,
    temperature=0,
    top_p=0.95,
    max_completion_tokens=int(os.environ.get("NVIDIA_MAX_TOKENS", "4096")),
)

llm_with_tools = llm.bind_tools(tools)

async def agent_node(state: AgentState):
    messages = state["messages"]
    
    # Calculate current workspace from message history
    workspace_content = []
    read_sections = set()
    
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("WORKSPACE_UPDATE::"):
            workspace_content.append(msg.content.split("WORKSPACE_UPDATE::", 1)[1].strip())
        
        # Track read history
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "read_section_content":
                    args = tc["args"]
                    pid = args.get("paper_id")
                    sec = args.get("section_title")
                    if pid and sec:
                        read_sections.add(f"- {pid} / {sec}")
    
    working_workspace = "\n\n".join(workspace_content) if workspace_content else ""
    read_history_str = "\n".join(sorted(read_sections)) if read_sections else "None"
    
    core_constraints = load_core_constraints()
    state["semantic_directives"] = core_constraints
    
    workspace = working_workspace if working_workspace else "No active workspace."
    
    system_prompt = f"""You are DeepScholar, an advanced academic research agent.

[CORE CONSTRAINTS]
{core_constraints}

[CURRENT WORKSPACE]
{workspace}

[READ HISTORY]
{read_history_str}

Please assist the user with their research inquiries.
For evidence gathering, prefer calling `hybrid_search` to retrieve relevant snippets. Only call `read_section_content` when you need full-section context.
"""
    
    # Use summary if available to reduce context
    if state.get("summary"):
        summary_msg = SystemMessage(content=f"[PREVIOUS CONVERSATION SUMMARY]:\n{state['summary']}")
        # Keep the last 10 messages for immediate context
        context_messages = [summary_msg] + messages[-10:]
    else:
        context_messages = messages

    response = await llm_with_tools.ainvoke([SystemMessage(content=system_prompt)] + context_messages)
    return {"messages": [response], "working_workspace": working_workspace}

async def compaction_node(state: AgentState):
    """
    Compresses history when token limit is reached using the LLM.
    """
    messages = state["messages"]
    current_summary = state.get("summary", "")
    
    # Summarize history except the last few messages
    msgs_to_summarize = messages[:-5]
    if not msgs_to_summarize:
        return {"messages": []}

    summary_prompt = f"""
    You are a context optimization assistant.
    Your task is to merge the new conversation history into the existing summary.
    
    [EXISTING SUMMARY]
    {current_summary}
    
    [NEW HISTORY TO MERGE]
    {msgs_to_summarize} 
    
    Please produce a single, dense paragraph summarizing the entire conversation history, preserving key facts, decisions, and workspace updates.
    """
    
    # Use a separate LLM call (or the same one)
    response = await llm.ainvoke([SystemMessage(content=summary_prompt)])
    new_summary = response.content
    
    return {"summary": new_summary, "messages": [SystemMessage(content=f"[System: Context compacted. Previous history summarized.]")]}

def router(state: AgentState) -> Literal["compaction", "tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
        
    if estimate_tokens(messages) > COMPACTION_THRESHOLD: 
        return "compaction"
    
    return "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("compaction", compaction_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {
        "compaction": "compaction",
        "tools": "tools",
        "__end__": END
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("compaction", "agent") 

app = workflow.compile()
