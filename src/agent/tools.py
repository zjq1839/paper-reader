import os
import json
import datetime
import math
from typing import Any, List, Dict, Optional
from langchain_core.tools import tool

DATA_DIR = os.path.join(os.getcwd(), "data", "processed")
MEMORY_DIR = os.path.join(os.getcwd(), "data", "memory")
HYBRID_INDEX_DB_PATH = os.environ.get("HYBRID_INDEX_DB_PATH", os.path.join(os.getcwd(), "data", "hybrid_index.db"))
HYBRID_VECTOR_DB_PATH = os.environ.get("HYBRID_VECTOR_DB_PATH", os.path.join(os.getcwd(), "data", "faiss_index"))

# Ensure memory directory exists
os.makedirs(MEMORY_DIR, exist_ok=True)

EPISODIC_MEMORY_FILE = os.path.join(MEMORY_DIR, "episodic_memory.jsonl")
SEMANTIC_KNOWLEDGE_FILE = os.path.join(MEMORY_DIR, "semantic_knowledge.md")

_HYBRID_INDEXER: Any = None

def _get_paper_dir(paper_id: str) -> str:
    return os.path.join(DATA_DIR, paper_id)

def _get_hybrid_indexer():
    global _HYBRID_INDEXER
    if _HYBRID_INDEXER is not None:
        return _HYBRID_INDEXER

    enable_vector = str(os.environ.get("HYBRID_VECTOR_ENABLED", "1")).strip().lower() not in ("0", "false", "no", "off")
    from src.ingestion.indexer import HybridIndexer

    _HYBRID_INDEXER = HybridIndexer(
        db_path=HYBRID_INDEX_DB_PATH,
        vector_db_path=HYBRID_VECTOR_DB_PATH,
        enable_vector=enable_vector,
    )
    return _HYBRID_INDEXER

@tool
def get_library_structure(paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get the list of available papers and their high-level outline (sections).
    If paper_id is provided, returns structure only for that paper.
    """
    library: List[Dict[str, Any]] = []
    if not os.path.exists(DATA_DIR):
        return []

    target_papers = [paper_id] if paper_id else os.listdir(DATA_DIR)

    for pid in target_papers:
        paper_path = _get_paper_dir(pid)
        if not os.path.exists(paper_path): 
            continue
            
        index_path = os.path.join(paper_path, "index.json")

        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                library.append(
                    {
                        "paper_id": pid,
                        "title": data.get("title", "Untitled"),
                        "sections": [s["title"] for s in data.get("sections", [])],
                    }
                )

    return library

@tool
def hybrid_search(query: str, k: int = 6, paper_id: Optional[str] = None, alpha: float = 0.7) -> str:
    """
    Search paper chunks using hybrid retrieval (vector + keyword fusion).
    Returns a JSON list of the most relevant chunk texts with metadata.
    """
    q = str(query or "").strip()
    if not q:
        return "[]"

    try:
        indexer = _get_hybrid_indexer()
    except Exception as e:
        return json.dumps({"error": f"Hybrid index not available: {e}"}, ensure_ascii=False)

    fetch_k = int(k) if int(k) > 0 else 6
    if paper_id:
        fetch_k = max(fetch_k * 4, fetch_k)

    try:
        docs = indexer.search(q, k=fetch_k, alpha=float(alpha))
    except Exception as e:
        return json.dumps({"error": f"Hybrid search failed: {e}"}, ensure_ascii=False)

    pid = str(paper_id or "").strip()
    if pid:
        docs = [d for d in docs if (d.metadata or {}).get("paper_id") == pid]

    out: List[Dict[str, Any]] = []
    max_chars = int(os.environ.get("HYBRID_SNIPPET_MAX_CHARS", "1400"))
    for d in docs[: int(k)]:
        md = d.metadata or {}
        content = str(d.page_content or "")
        if max_chars > 0 and len(content) > max_chars:
            content = content[: max_chars].rstrip() + "…"
        out.append(
            {
                "paper_id": md.get("paper_id"),
                "title": md.get("title"),
                "section_title": md.get("section_title"),
                "chunk_index": md.get("chunk_index"),
                "chunk_start_token": md.get("chunk_start_token"),
                "chunk_end_token": md.get("chunk_end_token"),
                "content": content,
                "source": md.get("source"),
            }
        )

    return json.dumps(out, ensure_ascii=False, indent=2)

@tool
def read_section_content(paper_id: str, section_title: str, page: int = 1, chars_per_page: int = 4000) -> str:
    """
    Read the content of a specific section from a paper.
    If content is long, it returns a specific page.
    page: Page number (1-based).
    chars_per_page: Number of characters per page (default 4000).
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
            content = f.read()
            
        total_chars = len(content)
        if total_chars <= chars_per_page:
            return content
            
        total_pages = math.ceil(total_chars / chars_per_page)
        
        if page < 1 or page > total_pages:
            return f"Error: Page {page} out of range (1-{total_pages})."
            
        start_idx = (page - 1) * chars_per_page
        end_idx = min(start_idx + chars_per_page, total_chars)
        
        chunk = content[start_idx:end_idx]
        
        return f"""[SECTION CONTENT - Page {page}/{total_pages}]
{chunk}

[SYSTEM NOTE]
This section is long ({total_chars} chars). You are viewing page {page} of {total_pages}.
To read the next part, call `read_section_content(paper_id="{paper_id}", section_title="{section_title}", page={page+1})`.
"""
            
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
