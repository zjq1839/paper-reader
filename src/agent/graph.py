import os
from typing import Annotated, Literal, TypedDict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent.tools import (
    get_library_structure,
    read_section_content,
    update_workspace,
    report_status,
    search_episodic_memory,
    update_semantic_knowledge
)

# Define State
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    working_workspace: str
    semantic_directives: str
    episodic_events: list[str]

# Load Core Constraints
CORE_FILE = "DEEPSCHOLAR_CORE.md"

def load_core_constraints():
    if os.path.exists(CORE_FILE):
        with open(CORE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    # Default constraints if file missing
    return """
# DEEPSCHOLAR CORE PROTOCOLS
1. Always prioritize evidence from the provided literature.
2. Use the hybrid retrieval system to find information.
3. Maintain a structured workspace in 'working_workspace'.
4. If conflicts arise between papers, use the Conflict Resolution Protocol.
"""

# Initialize Model & Tools
tools = [
    get_library_structure,
    read_section_content,
    update_workspace,
    report_status,
    search_episodic_memory,
    update_semantic_knowledge
]

# Model Selection Logic
nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "").strip().strip('"').strip("'")
if nvidia_api_key:
    llm = ChatNVIDIA(
        model=os.environ.get("NVIDIA_MODEL", "qwen/qwen3.5-397b-a17b"),
        api_key=nvidia_api_key,
        temperature=0,
        top_p=0.95,
        max_completion_tokens=4096 # Adjusted for safety
    )
else:
    llm = ChatOpenAI(model="gpt-4o", temperature=0) 

llm_with_tools = llm.bind_tools(tools)

# Define Nodes
def agent_node(state: AgentState):
    messages = state["messages"]
    
    # Load constraints dynamically (in case file changes)
    core_constraints = load_core_constraints()
    state["semantic_directives"] = core_constraints
    
    workspace = state.get("working_workspace", "No active workspace.")
    
    system_prompt = f"""You are DeepScholar, an advanced academic research agent.

[CORE CONSTRAINTS]
{core_constraints}

[CURRENT WORKSPACE]
{workspace}

Please assist the user with their research inquiries.
"""
    
    # We invoke the model. 
    # Note: We prepend the system message for this turn. 
    # In a real app, we might want to manage the system message more persistently in the list.
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)
    return {"messages": [response]}

def compaction_node(state: AgentState):
    """
    Compresses history when token limit is reached.
    """
    # Placeholder for compaction logic
    # In a real implementation, this would summarize the conversation and replace messages.
    # For now, we just append a marker.
    return {"messages": [SystemMessage(content="[System: Context compacted for memory optimization.]")]}

def router(state: AgentState) -> Literal["compaction", "tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # 1. If tool calls, MUST go to tools
    if last_message.tool_calls:
        return "tools"
        
    # 2. Check token count / message length (Simplified threshold: 50 messages)
    if len(messages) > 50: 
        return "compaction"
    
    # 3. Otherwise end
    return "__end__"

# Build Graph
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

# Compile
app = workflow.compile()
