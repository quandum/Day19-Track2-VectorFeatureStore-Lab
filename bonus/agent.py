"""HybridMemoryAgent — combines Vector Store (episodic memory) + Feature Store (user profile).

Usage:
    from bonus.agent import HybridMemoryAgent
    agent = HybridMemoryAgent()
    agent.remember("I read that Kubernetes auto-scaling is great for production")
    context = agent.recall("tell me about kubernetes")
    print(context)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

# ── Paths ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
FEAST_DIR = REPO_ROOT / "app" / "feast_repo"

# ── Constants ─────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
COLLECTION = "user_memories"
RRF_K = 60

Mode = Literal["keyword", "semantic", "hybrid"]


class HybridMemoryAgent:
    """A minimal POC agent combining episodic memory (vector) and user profile (Feast).

    Two main methods:
      - remember(text, user_id) — chunk → embed → upsert to Qdrant
      - recall(query, user_id) — hybrid search + Feast profile → assembled context
    """

    def __init__(self) -> None:
        # ── Embedder (lazy-loaded) ──
        self.embedder: TextEmbedding | None = None

        # ── Qdrant in-memory ──
        self.qdrant = QdrantClient(":memory:")
        try:
            self.qdrant.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
        except ValueError:
            pass  # already exists

        # ── BM25 index (per-user, rebuilt on each remember) ──
        self._bm25_index: dict[str, BM25Okapi] = {}
        self._bm25_texts: dict[str, list[str]] = {}
        self._bm25_meta: dict[str, list[dict]] = {}

        # ── Feast FeatureStore (lazy) ──
        self._feast = None

    # ── Properties ────────────────────────────────────────────
    @property
    def _embedder(self) -> TextEmbedding:
        if self.embedder is None:
            self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        return self.embedder

    @property
    def _feast_store(self):
        if self._feast is None:
            from feast import FeatureStore
            self._feast = FeatureStore(repo_path=str(FEAST_DIR))
        return self._feast

    # ── Tokenizer ─────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Whitespace tokenizer — same as main lab. Use underthesea in prod."""
        return text.lower().split()

    # ── Remember ──────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        # 1. Embed
        vec = list(self._embedder.embed([text]))[0]

        # 2. Upsert to Qdrant
        # Use a timestamp-based ID so each memory is unique
        memory_id = int(time.time() * 1000) % (2**63)
        self.qdrant.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=vec.tolist(),
                    payload={
                        "text": text,
                        "user_id": user_id,
                        "timestamp": time.time(),
                    },
                )
            ],
        )

        # 3. Update per-user BM25 index
        if user_id not in self._bm25_texts:
            self._bm25_texts[user_id] = []
            self._bm25_meta[user_id] = []
        self._bm25_texts[user_id].append(text)
        self._bm25_meta[user_id].append({"text": text, "user_id": user_id})
        tokenized = [self._tokenize(t) for t in self._bm25_texts[user_id]]
        self._bm25_index[user_id] = BM25Okapi(tokenized)

    # ── Recall ────────────────────────────────────────────────
    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile → return assembled context string."""
        top_k = 5
        depth = max(top_k * 5, 50)

        # 1. Hybrid search over this user's memories
        kw_ids = self._search_keyword(query, user_id, depth)
        sem_ids = self._search_semantic(query, user_id, depth)
        fused = self._rrf_fuse(kw_ids, sem_ids, top_k=top_k)

        # 2. Get user profile features from Feast online store
        profile_str = ""
        try:
            features = self._feast_store.get_online_features(
                features=[
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:preferred_language",
                    "user_profile_features:topic_affinity",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            profile_str = (
                f"User profile:\n"
                f"  - Topic affinity: {features.get('topic_affinity', ['?'])[0]}\n"
                f"  - Reading speed: {features.get('reading_speed_wpm', ['?'])[0]} wpm\n"
                f"  - Language: {features.get('preferred_language', ['?'])[0]}\n"
                f"  - Recent activity: {features.get('queries_last_hour', ['?'])[0]} queries in last hour\n"
                f"  - Topics explored (24h): {features.get('distinct_topics_24h', ['?'])[0]} topics\n"
            )
        except Exception as e:
            profile_str = f"(Feast unavailable: {e})\n"

        # 3. Assemble context
        memories_str = ""
        for i, mem in enumerate(fused, 1):
            memories_str += f"  {i}. {mem['text']}\n"

        context = (
            f"--- Context for user {user_id} ---\n"
            f"Query: {query!r}\n\n"
            f"{profile_str}\n"
            f"Top-{len(fused)} episodic memories:\n"
            f"{memories_str}\n"
            f"--- End context ---"
        )
        return context

    # ── Internal search methods ───────────────────────────────
    def _search_keyword(self, query: str, user_id: str, top_k: int) -> list[dict]:
        if user_id not in self._bm25_index:
            return []
        scores = self._bm25_index[user_id].get_scores(self._tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self._bm25_meta[user_id][i] for i in ranked]

    def _search_semantic(self, query: str, user_id: str, top_k: int) -> list[dict]:
        q_vec = next(self._embedder.embed([query])).tolist()
        hits = self.qdrant.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
        ).points
        return [{"text": p.payload["text"], "user_id": p.payload["user_id"]} for p in hits]

    def _rrf_fuse(self, kw: list[dict], sem: list[dict], top_k: int) -> list[dict]:
        rrf: dict[str, float] = {}
        meta: dict[str, dict] = {}
        for rank, item in enumerate(kw, start=1):
            tid = item["text"]
            rrf[tid] = rrf.get(tid, 0.0) + 1.0 / (RRF_K + rank)
            meta.setdefault(tid, item)
        for rank, item in enumerate(sem, start=1):
            tid = item["text"]
            rrf[tid] = rrf.get(tid, 0.0) + 1.0 / (RRF_K + rank)
            meta.setdefault(tid, item)
        ordered = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        return [meta[tid] for tid, _ in ordered]


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    agent = HybridMemoryAgent()
    agent.remember("Kubernetes auto-scaling helps production workloads scale horizontally")
    agent.remember("Cloud security requires encryption at rest and in transit")
    agent.remember("I prefer reading about AI and machine learning in Vietnamese")
    agent.remember("Deep learning models need GPU training for large datasets")
    agent.remember("FastAPI is great for building REST APIs with automatic docs")
    print(agent.recall("tell me about cloud security"))