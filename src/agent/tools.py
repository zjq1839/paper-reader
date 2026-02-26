import os
import json
import datetime
import math
from typing import Any, List, Dict, Optional
from langchain_core.tools import tool

DATA_DIR = os.path.join(os.getcwd(), "data", "processed")
MEMORY_DIR = os.path.join(os.getcwd(), "data", "memory")

# Ensure memory directory exists
os.makedirs(MEMORY_DIR, exist_ok=True)

EPISODIC_MEMORY_FILE = os.path.join(MEMORY_DIR, "episodic_memory.jsonl")
SEMANTIC_KNOWLEDGE_FILE = os.path.join(MEMORY_DIR, "semantic_knowledge.md")

def _get_paper_dir(paper_id: str) -> str:
    return os.path.join(DATA_DIR, paper_id)

@tool
def get_library_structure() -> List[Dict[str, Any]]:
    """
    Get the list of available papers and their high-level outline (sections).
    """
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
def read_section_content(paper_id: str, section_title: str) -> str:
    """
    Read the full content of a specific section from a paper.
    """
    paper_path = _get_paper_dir(paper_id)
    sections_dir = os.path.join(paper_path, "sections")
    
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

@tool
def search_episodic_memory(query: str, time_range: str = "all") -> str:
    """
    Search historical experimental reasoning records with time decay factor.
    query: The search query.
    time_range: 'all', 'recent' (last 24h), 'week' (last 7 days).
    """
    if not os.path.exists(EPISODIC_MEMORY_FILE):
        return "No episodic memory found."

    results = []
    now = datetime.datetime.now()
    
    with open(EPISODIC_MEMORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                timestamp = datetime.datetime.fromisoformat(record["timestamp"])
                
                # Time filtering
                if time_range == "recent" and (now - timestamp).total_seconds() > 86400:
                    continue
                if time_range == "week" and (now - timestamp).total_seconds() > 604800:
                    continue

                # Simple keyword match (replace with embedding search if needed later)
                if query.lower() in record["content"].lower():
                    # Calculate decay factor (Ebbinghaus-like: 1 / (1 + log(t+1)))
                    # t in hours
                    hours_diff = (now - timestamp).total_seconds() / 3600
                    decay = 1.0 / (1.0 + math.log(hours_diff + 1))
                    
                    results.append({
                        "content": record["content"],
                        "timestamp": record["timestamp"],
                        "score": decay
                    })
            except Exception:
                continue

    # Sort by score (decay factor implies recency/strength)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    if not results:
        return "No matching episodic memory found."
        
    return json.dumps(results[:5], indent=2, ensure_ascii=False)

@tool
def update_semantic_knowledge(topic: str, conclusion: str) -> str:
    """
    Write a distilled research pattern or key breakthrough to the global Tier 1 knowledge base.
    """
    entry = f"\n## {topic}\n- **Date:** {datetime.datetime.now().isoformat()}\n- **Conclusion:** {conclusion}\n"
    
    with open(SEMANTIC_KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
        
    return f"Successfully updated semantic knowledge for topic: {topic}"
