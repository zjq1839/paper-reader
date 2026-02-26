import os
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import chainlit as cl
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from typing import Any, Dict, Optional

from src.agent.lc_agent import agent

class ProgressCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        super().__init__()
        self._steps: Dict[str, cl.Step] = {}

    def _tool_display_name(self, tool_name: str, tool_input: Any) -> str:
        if isinstance(tool_input, dict):
            status = str(tool_input.get("status") or "").strip()
            if status:
                return status
        if isinstance(tool_input, str):
            status = tool_input.strip()
            if status:
                return status
        return "处理中"

    def _tool_result_summary(self, tool_name: str, output: Any) -> str:
        return ""

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: Optional[str] = None,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
        inputs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = str(serialized.get("name") or serialized.get("id") or "tool")
        tool_input: Any = inputs if inputs is not None else input_str
        if tool_name != "report_status":
            return
        step = cl.Step(name=self._tool_display_name(tool_name, tool_input), type="tool")
        await step.__aenter__()
        key = str(run_id) if run_id is not None else str(id(step))
        self._steps[key] = step

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id is not None else None
        if not key or key not in self._steps:
            return
        step = self._steps.pop(key)
        step.output = ""
        await step.__aexit__(None, None, None)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id is not None else None
        if not key or key not in self._steps:
            return
        step = self._steps.pop(key)
        step.output = f"失败：{type(error).__name__}"
        await step.__aexit__(None, None, None)

def _normalize_history(items):
    normalized = []
    for it in items or []:
        if isinstance(it, dict):
            role = (it.get("role") or "").lower()
            content = it.get("content") or ""
            if role == "system":
                normalized.append(SystemMessage(content=content))
            elif role == "assistant":
                normalized.append(AIMessage(content=content))
            else:
                normalized.append(HumanMessage(content=content))
        else:
            normalized.append(it)
    return normalized

def _truncate_text(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n（内容过长，已截断）"

@cl.on_chat_start
async def start():
    cl.user_session.set("workspace", "")
    welcome = cl.Message(content="DeepScholar 已就绪。")
    await welcome.send()
    cl.user_session.set("workspace_for_id", welcome.id)
    
    # Initialize workspace side view
    await update_workspace_ui("")

async def update_workspace_ui(content: str):
    """
    Update the side view with the current workspace content.
    """
    if not content:
        content = "*Workspace is empty.*"
    try:
        await cl.ElementSidebar.set_title("Research Workspace")
        await cl.ElementSidebar.set_elements([cl.Text(name="Research Workspace", content=content)])
    except Exception:
        for_id = cl.user_session.get("workspace_for_id")
        if not for_id:
            return
        await cl.Text(name="Research Workspace", content=content, display="inline").send(for_id=for_id)

@cl.on_message
async def main(message: cl.Message):
    # Get current state from session or just maintain history via LangGraph?
    # LangGraph is stateful. We should maintain the state or just pass messages.
    # For this simple MVP, we'll let LangGraph handle the state but we need to persist it 
    # or pass the full history. LangGraph's "messages" key usually expects the full history 
    # or the new messages if using a checkpointer.
    # Since we didn't set up a checkpointer, we need to pass the conversation history?
    # Wait, StateGraph with `Annotated[List[BaseMessage], operator.add]` means we can pass new messages 
    # and it appends. But `app_graph.invoke` starts a NEW execution.
    # If we want conversation memory, we need to pass the accumulated history or use a checkpointer.
    
    # Let's use a checkpointer for simplicity if we can, or just manage history manually in session.
    # Managing manually in session:
    history = _normalize_history(cl.user_session.get("history", []))
    history.append(HumanMessage(content=message.content))
    
    # Run the graph
    # We'll stream the output to show intermediate steps
    
    config = RunnableConfig(callbacks=[ProgressCallbackHandler()])
    res = await agent.ainvoke({"messages": history}, config=config)

    updated_messages = res["messages"]
    cl.user_session.set("history", updated_messages)

    workspace_updates = []
    for msg in updated_messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("WORKSPACE_UPDATE::"):
            workspace_updates.append(msg.content.split("WORKSPACE_UPDATE::", 1)[1].strip())

    if workspace_updates:
        current_workspace = cl.user_session.get("workspace", "")
        merged = (current_workspace + "\n\n" + "\n\n".join(workspace_updates)).strip() if current_workspace else "\n\n".join(workspace_updates).strip()
        cl.user_session.set("workspace", merged)
        await update_workspace_ui(merged)

    final_content = ""
    for msg in reversed(updated_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            final_content = msg.content.strip()
            break

    if final_content:
        current_workspace = str(cl.user_session.get("workspace", "") or "").strip()
        if not current_workspace:
            if ("工作区" in final_content) or ("workspace" in final_content.lower()):
                cl.user_session.set("workspace", _truncate_text(final_content))
                await update_workspace_ui(_truncate_text(final_content))
        await cl.Message(content=final_content).send()
    else:
        await cl.Message(content="（本轮模型没有返回可展示的文本内容）").send()


