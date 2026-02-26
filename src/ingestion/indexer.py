import os
import sqlite3
import json
from typing import List, Any, Optional, Tuple

import numpy as np
import faiss
from openai import OpenAI
from langchain_core.documents import Document


class NvidiaBgeM3Embeddings:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "baai/bge-m3",
    ):
        resolved_api_key = (api_key or os.environ.get("NVIDIA_API_KEY", "")).strip().strip('"').strip("'")
        if not resolved_api_key:
            raise RuntimeError("NVIDIA_API_KEY 未设置，无法生成 embedding")
        resolved_base_url = (base_url or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")).strip()
        resolved_base_url = resolved_base_url.strip("`").strip().strip('"').strip("'")

        self._client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self._model = (os.environ.get("NVIDIA_EMBED_MODEL", model) or model).strip()

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        clean = [t if isinstance(t, str) else str(t) for t in (texts or [])]
        clean = [t.strip() for t in clean if isinstance(t, str) and t.strip()]
        if not clean:
            return np.zeros((0, 0), dtype=np.float32)
        res = self._client.embeddings.create(
            input=clean,
            model=self._model,
            encoding_format="float",
            extra_body={"truncate": "NONE"},
        )
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in res.data]
        return np.stack(vectors, axis=0)

    def embed_query(self, text: str) -> np.ndarray:
        mat = self.embed_documents([text])
        return mat[0]


class LocalFaissStore:
    def __init__(self, dir_path: str, embeddings: NvidiaBgeM3Embeddings):
        self._dir_path = dir_path
        self._embeddings = embeddings
        self._index_path = os.path.join(dir_path, "index.faiss")
        self._docstore_path = os.path.join(dir_path, "docstore.json")
        self._index: Optional[faiss.Index] = None
        self._docs: List[Document] = []
        os.makedirs(dir_path, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._docstore_path):
            with open(self._docstore_path, "r", encoding="utf-8") as f:
                raw = json.load(f) or []
            self._docs = [Document(page_content=item["page_content"], metadata=item.get("metadata") or {}) for item in raw]

        if os.path.exists(self._index_path):
            self._index = faiss.read_index(self._index_path)

    def _save(self) -> None:
        if self._index is not None:
            faiss.write_index(self._index, self._index_path)
        payload = [{"page_content": d.page_content, "metadata": d.metadata} for d in self._docs]
        with open(self._docstore_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        kept: List[Document] = [d for d in documents if isinstance(d.page_content, str) and d.page_content.strip()]
        if not kept:
            return
        texts = [d.page_content for d in kept]
        vecs = self._embeddings.embed_documents(texts)
        if vecs.size == 0:
            return

        faiss.normalize_L2(vecs)
        dim = int(vecs.shape[1])
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)
        self._docs.extend(kept)
        self._save()

    def similarity_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if self._index is None or not self._docs:
            return []
        q = self._embeddings.embed_query(query).astype(np.float32)
        q = np.expand_dims(q, axis=0)
        faiss.normalize_L2(q)
        scores, idxs = self._index.search(q, k)
        out: List[Tuple[Document, float]] = []
        for i, score in zip(idxs[0].tolist(), scores[0].tolist()):
            if i < 0 or i >= len(self._docs):
                continue
            out.append((self._docs[i], float(score)))
        return out

class HybridIndexer:
    def __init__(self, db_path: str = "data/hybrid_index.db", vector_db_path: str = "data/faiss_index"):
        self.db_path = db_path
        self.vector_db_path = vector_db_path
        self.embeddings = NvidiaBgeM3Embeddings()
        self.vector_store = LocalFaissStore(self.vector_db_path, self.embeddings)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)

        self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite database with FTS5 for BM25."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Create FTS5 table
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                content,
                metadata,
                tokenize='porter'
            )
        ''')
        conn.commit()
        conn.close()

    def add_documents(self, documents: List[Document]):
        """Add documents to both SQLite FTS and FAISS vector store."""
        if not documents:
            return
        kept: List[Document] = [d for d in documents if isinstance(d.page_content, str) and d.page_content.strip()]
        if not kept:
            return

        # 1. Add to SQLite FTS
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for doc in kept:
            cursor.execute(
                "INSERT INTO documents_fts (content, metadata) VALUES (?, ?)",
                (doc.page_content, json.dumps(doc.metadata))
            )
        conn.commit()
        conn.close()

        # 2. Add to Vector Store
        self.vector_store.add_documents(kept)

    def search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Document]:
        """
        Hybrid search using fusion of Vector (Cosine) and Keyword (BM25/FTS).
        alpha: Weight for vector search (default 0.7). 1.0 = pure vector, 0.0 = pure keyword.
        """
        # 1. Vector Search
        vector_results = []
        vector_results_raw = self.vector_store.similarity_search_with_score(query, k=k * 2)
        vector_results = vector_results_raw

        # 2. Keyword Search (SQLite FTS)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content, metadata, bm25(documents_fts) AS score
            FROM documents_fts 
            WHERE documents_fts MATCH ? 
            ORDER BY score 
            LIMIT ?
        """, (query, k*2))
        
        rows = cursor.fetchall()
        conn.close()
        
        doc_map = {}

        for rank, (doc, score) in enumerate(vector_results):
            doc_id = hash(doc.page_content)
            if doc_id not in doc_map:
                doc_map[doc_id] = {"doc": doc, "vec_rank": rank + 1, "fts_rank": 1000}
            else:
                doc_map[doc_id]["vec_rank"] = rank + 1

        for rank, (content, metadata_json, score) in enumerate(rows):
            metadata = json.loads(metadata_json)
            doc = Document(page_content=content, metadata=metadata)
            doc_id = hash(content)
            if doc_id not in doc_map:
                doc_map[doc_id] = {"doc": doc, "vec_rank": 1000, "fts_rank": rank + 1}
            else:
                doc_map[doc_id]["fts_rank"] = rank + 1

        fused_results = []
        for doc_id, data in doc_map.items():
            rrf_score = alpha * (1.0 / (60 + data["vec_rank"])) + (1.0 - alpha) * (1.0 / (60 + data["fts_rank"]))
            fused_results.append((data['doc'], rrf_score))

        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in fused_results[:k]]

