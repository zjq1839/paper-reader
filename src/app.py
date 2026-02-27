import os
import sys
import time
import uuid
import json
import datetime
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
from chainlit.config import config as chainlit_config
from chainlit.utils import wrap_user_function
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from typing import Any, Dict, Optional

from src.agent.lc_agent import agent

TRACE_DIR = os.path.join(project_root, "data", "memory", "traces")

def _truncate_text(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n（内容过长，已截断）"

def _format_hms(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))

def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def _summarize_output(tool_name: str, output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        s = output.strip()
        if tool_name == "update_workspace":
            if s.startswith("WORKSPACE_UPDATE::"):
                payload = s.split("WORKSPACE_UPDATE::", 1)[1].strip()
                return f"写入 Workspace（{len(payload)} 字）"
            return "写入 Workspace"
        if tool_name == "report_status":
            try:
                parsed = json.loads(s)
                status = str(parsed.get("status") or "").strip()
                if status:
                    return status
            except Exception:
                pass
            return s[:200]
        if len(s) <= 260:
            return s
        return f"{s[:240].rstrip()}…（{len(s)} 字）"
    return _safe_json_dumps(output)[:260]

async def _persist_trace_event(trace_session_id: str, event: Dict[str, Any]) -> None:
    try:
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, f"{trace_session_id}.jsonl")
        line = json.dumps(event, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return

async def _render_sidebar(workspace: str, trace_md: str) -> None:
    ws = (workspace or "").strip()
    tr = (trace_md or "").strip()
    if not ws:
        ws = "*Workspace is empty.*"
    if not tr:
        tr = "*Trace is empty.*"
    MarkdownEl = None
    try:
        MarkdownEl = cl.Markdown
    except Exception:
        MarkdownEl = None
    WorkspaceEl = MarkdownEl or cl.Text
    try:
        await cl.ElementSidebar.set_title("Research Workspace")
        await cl.ElementSidebar.set_elements(
            [
                WorkspaceEl(name="Research Workspace", content=ws),
                WorkspaceEl(name="Agent Trace", content=tr),
            ]
        )
    except Exception:
        for_id = cl.user_session.get("workspace_for_id")
        if not for_id:
            return
        try:
            await WorkspaceEl(name="Research Workspace", content=ws, display="inline").send(for_id=for_id)
            await WorkspaceEl(name="Agent Trace", content=tr, display="inline").send(for_id=for_id)
        except TypeError:
            await WorkspaceEl(name="Research Workspace", content=ws).send(for_id=for_id)
            await WorkspaceEl(name="Agent Trace", content=tr).send(for_id=for_id)

def _read_jsonl(path: str, max_lines: int = 5000) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        items.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return items

def _extract_workspace_from_events(events: list[Dict[str, Any]]) -> str:
    parts: list[str] = []
    for ev in events or []:
        if str(ev.get("type") or "").strip() != "tool_start":
            continue
        if str(ev.get("name") or "").strip() != "update_workspace":
            continue
        tool_input = ev.get("input")
        payload = ""
        if isinstance(tool_input, dict):
            payload = str(tool_input.get("content") or "").strip()
        elif isinstance(tool_input, str):
            payload = tool_input.strip()
        if payload:
            parts.append(payload)
    return "\n\n".join(parts).strip()

def _pick_trace_session_id(prefer: Optional[str] = None, created_at: Optional[str] = None) -> Optional[str]:
    if prefer:
        candidate = os.path.join(TRACE_DIR, f"{prefer}.jsonl")
        if os.path.exists(candidate):
            return prefer

    try:
        if not os.path.isdir(TRACE_DIR):
            return None
        files = [os.path.join(TRACE_DIR, p) for p in os.listdir(TRACE_DIR) if p.endswith(".jsonl")]
        if not files:
            return None
        created_ts: Optional[float] = None
        if created_at:
            try:
                created_ts = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            except Exception:
                created_ts = None
        if created_ts is not None:
            files.sort(key=lambda p: (abs(os.path.getmtime(p) - created_ts), -os.path.getmtime(p)))
        else:
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return Path(files[0]).stem
    except Exception:
        return None

def _trace_markdown(events: list[Dict[str, Any]], limit: int = 60) -> str:
    items = (events or [])[-limit:]
    if not items:
        return ""
    lines: list[str] = []
    for ev in items:
        ts = ev.get("ts", 0.0)
        etype = str(ev.get("type") or "").strip()
        name = str(ev.get("name") or "").strip()
        status = str(ev.get("status") or "").strip()
        dur_ms = ev.get("duration_ms")
        dur_txt = f"{int(dur_ms)}ms" if isinstance(dur_ms, (int, float)) else ""
        parts = [p for p in [etype, name, status, dur_txt] if p]
        lines.append(f"- [{_format_hms(float(ts) if ts else time.time())}] " + " · ".join(parts))
        summary = str(ev.get("summary") or "").strip()
        if summary:
            lines.append(f"  - {summary}")
    return "\n".join(lines)

class ProgressCallbackHandler(AsyncCallbackHandler):
    def __init__(self, trace_session_id: str):
        super().__init__()
        self._steps: Dict[str, cl.Step] = {}
        self._starts: Dict[str, float] = {}
        self._trace_session_id = trace_session_id

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

    def _tool_display_input(self, tool_name: str, tool_input: Any) -> str:
        if not tool_input:
            return ""
        if isinstance(tool_input, dict):
            if tool_name == "hybrid_search":
                query = str(tool_input.get("query") or "").strip()
                paper_id = str(tool_input.get("paper_id") or "").strip()
                k = tool_input.get("k")
                alpha = tool_input.get("alpha")
                parts = []
                if query:
                    parts.append(f"query={query}")
                if paper_id:
                    parts.append(f"paper_id={paper_id}")
                if k is not None:
                    parts.append(f"k={k}")
                if alpha is not None:
                    parts.append(f"alpha={alpha}")
                if parts:
                    return _truncate_text("\n".join(parts), 800)
            if tool_name == "read_section_content":
                paper_id = str(tool_input.get("paper_id") or "").strip()
                section_title = str(tool_input.get("section_title") or "").strip()
                if paper_id or section_title:
                    return _truncate_text(f"paper_id={paper_id}\nsection_title={section_title}", 800)
            if tool_name == "update_workspace":
                content = str(tool_input.get("content") or "").strip()
                if content:
                    return _truncate_text(content, 800)
            if tool_name == "search_episodic_memory":
                query = str(tool_input.get("query") or "").strip()
                time_range = str(tool_input.get("time_range") or "").strip()
                return _truncate_text(f"query={query}\ntime_range={time_range}", 800)
            if tool_name == "update_semantic_knowledge":
                topic = str(tool_input.get("topic") or "").strip()
                conclusion = str(tool_input.get("conclusion") or "").strip()
                return _truncate_text(f"topic={topic}\nconclusion={conclusion}", 800)
            if tool_name == "report_status":
                status = str(tool_input.get("status") or "").strip()
                return _truncate_text(status, 800)
            return _truncate_text(_safe_json_dumps(tool_input), 800)
        if isinstance(tool_input, str):
            return _truncate_text(tool_input, 800)
        return _truncate_text(str(tool_input), 800)

    async def _append_trace(self, ev: Dict[str, Any]) -> None:
        events = cl.user_session.get("trace_events", []) or []
        events.append(ev)
        cl.user_session.set("trace_events", events)
        await _persist_trace_event(self._trace_session_id, ev)
        workspace = cl.user_session.get("workspace", "") or ""
        await _render_sidebar(workspace, _trace_markdown(events))

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
        step_name = self._tool_display_name(tool_name, tool_input) if tool_name == "report_status" else f"调用工具：{tool_name}"
        step = cl.Step(name=step_name, type="tool")
        step.input = self._tool_display_input(tool_name, tool_input)
        await step.__aenter__()
        key = str(run_id) if run_id is not None else str(id(step))
        self._steps[key] = step
        self._starts[key] = time.time()
        await self._append_trace(
            {
                "ts": self._starts[key],
                "type": "tool_start",
                "name": tool_name,
                "status": step_name,
                "input": tool_input,
            }
        )

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
        started = self._starts.pop(key, None)
        duration_ms = (time.time() - started) * 1000 if started else None
        tool_name = step.name.replace("调用工具：", "").strip() if step.name.startswith("调用工具：") else "report_status"
        step.output = _truncate_text(_summarize_output(tool_name, output), 1200)
        await step.__aexit__(None, None, None)
        await self._append_trace(
            {
                "ts": time.time(),
                "type": "tool_end",
                "name": tool_name,
                "duration_ms": duration_ms,
                "summary": _summarize_output(tool_name, output),
            }
        )

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
        started = self._starts.pop(key, None)
        duration_ms = (time.time() - started) * 1000 if started else None
        step.output = f"失败：{type(error).__name__}"
        await step.__aexit__(None, None, None)
        tool_name = step.name.replace("调用工具：", "").strip() if step.name.startswith("调用工具：") else "report_status"
        await self._append_trace(
            {
                "ts": time.time(),
                "type": "tool_error",
                "name": tool_name,
                "duration_ms": duration_ms,
                "summary": f"{type(error).__name__}: {str(error)[:200]}",
            }
        )

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        step = cl.Step(name="思考中", type="llm")
        await step.__aenter__()
        key = str(run_id) if run_id is not None else str(id(step))
        self._steps[key] = step
        self._starts[key] = time.time()
        await self._append_trace(
            {
                "ts": self._starts[key],
                "type": "llm_start",
                "name": str(serialized.get("name") or "llm"),
            }
        )

    async def on_llm_end(
        self,
        response: Any,
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
        started = self._starts.pop(key, None)
        duration_ms = (time.time() - started) * 1000 if started else None
        step.output = ""
        await step.__aexit__(None, None, None)
        await self._append_trace(
            {
                "ts": time.time(),
                "type": "llm_end",
                "name": "llm",
                "duration_ms": duration_ms,
            }
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id) if run_id is not None else None
        if key and key in self._steps:
            step = self._steps.pop(key)
            started = self._starts.pop(key, None)
            duration_ms = (time.time() - started) * 1000 if started else None
            step.output = f"失败：{type(error).__name__}"
            await step.__aexit__(None, None, None)
            await self._append_trace(
                {
                    "ts": time.time(),
                    "type": "llm_error",
                    "name": "llm",
                    "duration_ms": duration_ms,
                    "summary": f"{type(error).__name__}: {str(error)[:200]}",
                }
            )

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
        elif isinstance(it, ToolMessage):
            raw = it.content
            if isinstance(raw, str):
                normalized.append(it)
                continue
            fixed = "" if raw is None else _safe_json_dumps(raw)
            tool_call_id = getattr(it, "tool_call_id", None)
            name = getattr(it, "name", None)
            if tool_call_id:
                kwargs = {"tool_call_id": tool_call_id}
                if name:
                    kwargs["name"] = name
                try:
                    normalized.append(ToolMessage(content=fixed, **kwargs))
                except Exception:
                    normalized.append(HumanMessage(content=fixed))
            else:
                normalized.append(HumanMessage(content=fixed))
        else:
            normalized.append(it)
    return normalized

@cl.on_chat_start
async def start():
    cl.user_session.set("workspace", "")
    cl.user_session.set("trace_session_id", str(uuid.uuid4()))
    cl.user_session.set("trace_events", [])
    welcome = cl.Message(content="DeepScholar 已就绪。")
    await welcome.send()
    cl.user_session.set("workspace_for_id", welcome.id)
    
    auto_hydrate = str(os.environ.get("AUTO_HYDRATE_SIDEBAR", "0") or "").strip().lower() in ["1", "true", "yes", "y"]
    if auto_hydrate:
        hydrated_trace_session_id = _pick_trace_session_id()
        if hydrated_trace_session_id:
            hydrated_events = _read_jsonl(os.path.join(TRACE_DIR, f"{hydrated_trace_session_id}.jsonl"))
            hydrated_workspace = _extract_workspace_from_events(hydrated_events)
            cl.user_session.set("workspace", hydrated_workspace)
            cl.user_session.set("trace_events", hydrated_events)
            await _render_sidebar(hydrated_workspace, _trace_markdown(hydrated_events))
            return

    await _render_sidebar("", "")

@cl.on_chat_resume
async def resume(thread: Dict[str, Any]):
    trace_session_id = _pick_trace_session_id(
        prefer=str((thread.get("metadata") or {}).get("trace_session_id") or "").strip() or None,
        created_at=str(thread.get("createdAt") or "").strip() or None,
    )
    if trace_session_id:
        events = _read_jsonl(os.path.join(TRACE_DIR, f"{trace_session_id}.jsonl"))
        workspace = _extract_workspace_from_events(events)
        cl.user_session.set("trace_session_id", trace_session_id)
        cl.user_session.set("trace_events", events)
        cl.user_session.set("workspace", workspace)
        await _render_sidebar(workspace, _trace_markdown(events))
    else:
        await _render_sidebar(str(cl.user_session.get("workspace", "") or ""), _trace_markdown(cl.user_session.get("trace_events", []) or []))

@cl.on_message
async def main(message: cl.Message):
    cmd = str(message.content or "").strip()
    cmd_l = cmd.lower()
    if cmd_l in ["/sidebar", "/trace", "/workspace", "/ws"]:
        ws = str(cl.user_session.get("workspace", "") or "").strip() or "*Workspace is empty.*"
        tr = _trace_markdown(cl.user_session.get("trace_events", []) or []) or "*Trace is empty.*"
        await cl.Message(content=f"## Workspace\n\n{ws}\n\n## Trace\n\n{tr}").send()
        return
    if cmd_l in ["/hydrate", "/resume"]:
        trace_session_id = _pick_trace_session_id()
        if trace_session_id:
            events = _read_jsonl(os.path.join(TRACE_DIR, f"{trace_session_id}.jsonl"))
            workspace = _extract_workspace_from_events(events)
            cl.user_session.set("trace_session_id", trace_session_id)
            cl.user_session.set("trace_events", events)
            cl.user_session.set("workspace", workspace)
            await _render_sidebar(workspace, _trace_markdown(events))
            await cl.Message(content=f"已从 traces 恢复：{trace_session_id}").send()
        else:
            await cl.Message(content="未找到可恢复的 traces/*.jsonl").send()
        return
    if cmd_l in ["/clear", "/reset", "/clearws"]:
        cl.user_session.set("workspace", "")
        cl.user_session.set("trace_events", [])
        await _render_sidebar("", "")
        await cl.Message(content="已清空 Workspace/Trace。").send()
        return
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
    
    cl.user_session.set("trace_events", [])
    trace_session_id = cl.user_session.get("trace_session_id") or str(uuid.uuid4())
    config = RunnableConfig(callbacks=[ProgressCallbackHandler(str(trace_session_id))])
    res = await agent.ainvoke({"messages": history}, config=config)

    updated_messages = _normalize_history(res["messages"])
    cl.user_session.set("history", updated_messages)

    workspace_updates = []
    for msg in updated_messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("WORKSPACE_UPDATE::"):
            workspace_updates.append(msg.content.split("WORKSPACE_UPDATE::", 1)[1].strip())

    if workspace_updates:
        merged = "\n\n".join(workspace_updates).strip()
        cl.user_session.set("workspace", merged)
        events = cl.user_session.get("trace_events", []) or []
        await _render_sidebar(merged, _trace_markdown(events))

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
                events = cl.user_session.get("trace_events", []) or []
                await _render_sidebar(_truncate_text(final_content), _trace_markdown(events))
        await cl.Message(content=final_content).send()
    else:
        await cl.Message(content="（本轮模型没有返回可展示的文本内容）").send()

async def _safe_dispatch_message(message):
    if message is None:
        return
    await main(message)

try:
    chainlit_config.code.on_message = wrap_user_function(_safe_dispatch_message)
except Exception:
    pass

