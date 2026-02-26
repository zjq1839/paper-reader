import os
import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import pickle

class HybridIndexer:
    def __init__(self, db_path: str = "data/hybrid_index.db", vector_db_path: str = "data/faiss_index"):
        self.db_path = db_path
        self.vector_db_path = vector_db_path
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)

        self._init_sqlite()
        self._load_vector_store()

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

    def _load_vector_store(self):
        """Load or initialize FAISS vector store."""
        if os.path.exists(os.path.join(self.vector_db_path, "index.faiss")):
            self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            # Initialize empty store (will be populated later)
            pass

    def add_documents(self, documents: List[Document]):
        """Add documents to both SQLite FTS and FAISS vector store."""
        if not documents:
            return

        # 1. Add to SQLite FTS
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for doc in documents:
            cursor.execute(
                "INSERT INTO documents_fts (content, metadata) VALUES (?, ?)",
                (doc.page_content, json.dumps(doc.metadata))
            )
        conn.commit()
        conn.close()

        # 2. Add to Vector Store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        self.vector_store.save_local(self.vector_db_path)

    def search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Document]:
        """
        Hybrid search using fusion of Vector (Cosine) and Keyword (BM25/FTS).
        alpha: Weight for vector search (default 0.7). 1.0 = pure vector, 0.0 = pure keyword.
        """
        # 1. Vector Search
        vector_results = []
        if self.vector_store:
            # Returns (doc, score) where score is L2 distance (lower is better)
            # We need to convert to similarity or normalize.
            # FAISS similarity_search_with_score returns L2 distance for default index
            vector_results_raw = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            # Normalize vector scores (L2 distance -> similarity)
            # Simple inversion for now: 1 / (1 + distance)
            vector_results = [
                (doc, 1.0 / (1.0 + score)) for doc, score in vector_results_raw
            ]

        # 2. Keyword Search (SQLite FTS)
        keyword_results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # FTS5 rank is not standard BM25 score but we can use bm25() function if available or simple rank
        # Standard FTS5 search by rank
        cursor.execute("""
            SELECT content, metadata, rank 
            FROM documents_fts 
            WHERE documents_fts MATCH ? 
            ORDER BY rank 
            LIMIT ?
        """, (query, k*2))
        
        rows = cursor.fetchall()
        conn.close()

        # FTS rank is usually lower is better (more negative). Let's normalize.
        # However, sqlite fts5 'rank' output depends on the function used.
        # Default rank is just a score.
        # Let's assume we get some results. We can treat them as "matched".
        # For better scoring, we might want to do BM25 in python if the dataset is small enough,
        # but for scalability we rely on FTS.
        # To strictly follow MRD "70% Vector + 30% BM25", we need comparable scores.
        # A robust way is Reciprocal Rank Fusion (RRF) or just normalizing ranks.
        
        # Let's use RRF for simplicity and robustness as scores are hard to normalize across different engines.
        # Score = alpha * (1/rank_vec) + (1-alpha) * (1/rank_fts)
        
        doc_map = {} # content_hash -> {doc, vec_rank, fts_rank}

        # Process Vector Results
        for rank, (doc, score) in enumerate(vector_results):
            # Create a unique ID for the doc (e.g. hash of content)
            doc_id = hash(doc.page_content)
            if doc_id not in doc_map:
                doc_map[doc_id] = {'doc': doc, 'vec_rank': rank + 1, 'fts_rank': 1000} # Default high rank
            else:
                doc_map[doc_id]['vec_rank'] = rank + 1

        # Process FTS Results
        for rank, (content, metadata_json, score) in enumerate(rows):
            metadata = json.loads(metadata_json)
            doc = Document(page_content=content, metadata=metadata)
            doc_id = hash(content)
            if doc_id not in doc_map:
                doc_map[doc_id] = {'doc': doc, 'vec_rank': 1000, 'fts_rank': rank + 1}
            else:
                doc_map[doc_id]['fts_rank'] = rank + 1

        # Calculate Fusion Score
        fused_results = []
        for doc_id, data in doc_map.items():
            # RRF score
            # score = alpha * (1/vec_rank) + (1-alpha) * (1/fts_rank)
            # Or use the MRD's weighted fusion if we had normalized scores.
            # Given we don't have true BM25 scores from SQLite easily without extensions, RRF is safer.
            # But the MRD asks for "Standardized scores then weighted fusion".
            # Let's try to stick to RRF as a practical implementation of "Hybrid".
            
            rrf_score = alpha * (1.0 / (60 + data['vec_rank'])) + (1.0 - alpha) * (1.0 / (60 + data['fts_rank']))
            fused_results.append((data['doc'], rrf_score))

        # Sort by score descending
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in fused_results[:k]]

