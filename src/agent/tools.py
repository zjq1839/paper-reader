import os
import json
from typing import Any, List, Dict, Optional
try:
    from langchain_core.tools import tool
except Exception:
    class _FallbackTool:
        def __init__(self, func):
            self._func = func
            self.__name__ = getattr(func, "__name__", "tool")
            self.__doc__ = getattr(func, "__doc__", None)

        def invoke(self, args: Optional[Dict[str, Any]] = None) -> Any:
            if args is None:
                args = {}
            return self._func(**args)

        def __call__(self, *args, **kwargs):
            return self._func(*args, **kwargs)

    def tool(func):
        return _FallbackTool(func)

DATA_DIR = os.path.join(os.getcwd(), "data", "processed")

def _get_paper_dir(paper_id: str) -> str:
    return os.path.join(DATA_DIR, paper_id)

def get_library_structure_impl() -> List[Dict[str, Any]]:
    library: List[Dict[str, Any]] = []
    if not os.path.exists(DATA_DIR):
        return []

    for paper_id in os.listdir(DATA_DIR):
        paper_path = _get_paper_dir(paper_id)
        index_path = os.path.join(paper_path, "index.json")

        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                library.append(
                    {
                        "paper_id": paper_id,
                        "title": data.get("title", "Untitled"),
                        "sections": [s["title"] for s in data.get("sections", [])],
                    }
                )

    return library

@tool
def get_library_structure() -> List[Dict[str, Any]]:
    """
    Get the list of available papers and their high-level outline (sections).
    """
    return get_library_structure_impl()

@tool
def read_section_content(paper_id: str, section_title: str) -> str:
    """
    Read the full content of a specific section from a paper.
    """
    paper_path = _get_paper_dir(paper_id)
    sections_dir = os.path.join(paper_path, "sections")
    
    # We need to find the file corresponding to the section title
    # This assumes we save sections with titles as filenames or use an index.
    # For MVP, let's look up the filename from index.json
    index_path = os.path.join(paper_path, "index.json")
    if not os.path.exists(index_path):
        return "Error: Paper index not found."

    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    target_filename = None
    for sec in data.get("sections", []):
        if sec["title"] == section_title:
            target_filename = sec.get("filename")
            break
            
    if not target_filename:
        return f"Error: Section '{section_title}' not found in paper '{paper_id}'."
        
    section_path = os.path.join(sections_dir, target_filename)
    if os.path.exists(section_path):
        with open(section_path, "r", encoding="utf-8") as f:
            return f.read()
            
    return "Error: Section content file not found."

@tool
def update_workspace(content: str) -> str:
    """
    Add a note or insight to the research workspace.
    This content will be visible in the side panel.
    """
    return f"WORKSPACE_UPDATE::{content}"

@tool
def report_status(status: str) -> str:
    """
    Emit a short progress update for the UI.
    The model should call this before other tools, with a short Chinese phrase.
    """
    return json.dumps({"type": "status", "status": status}, ensure_ascii=False)
