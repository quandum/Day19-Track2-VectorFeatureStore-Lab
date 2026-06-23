# Hybrid Memory Agent — Architecture Document

**Học viên:** Trần Mạnh Chánh Quân — **Mã học viên:** 2A202600786
**Cohort:** A20-K2
**Lab:** Day 19 — Bonus Challenge

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     User Query                                │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│                 HybridMemoryAgent.recall()                    │
│                                                              │
│   ┌──────────────┐    ┌──────────────────┐                   │
│   │  Vector Store │    │  Feature Store   │                   │
│   │  (Qdrant)     │    │  (Feast/SQLite)  │                   │
│   │               │    │                  │                   │
│   │  Episodic     │    │  User Profile:   │                   │
│   │  Memory:      │    │  - topic_affinity│                   │
│   │  - past docs  │    │  - reading_speed │                   │
│   │  - notes      │    │  - preferred_lang│                   │
│   │  - convos     │    │                  │                   │
│   │               │    │  Activity:       │                   │
│   │  Hybrid       │    │  - queries_last_ │                   │
│   │  Search:      │    │    hour          │                   │
│   │  BM25+Vector  │    │  - distinct_     │                   │
│   │  + RRF k=60   │    │    topics_24h    │                   │
│   └──────┬───────┘    └────────┬─────────┘                   │
│          │                     │                             │
│          └──────────┬──────────┘                             │
│                     ▼                                        │
│          ┌────────────────────┐                              │
│          │  Context Assembly  │                              │
│          │  "User likes X...  │                              │
│          │   Top memories: Y" │                              │
│          └────────────────────┘                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │  Return context │
                 │  string (no LLM)│
                 └─────────────────┘
```

**Data Flow:**
1. User sends a query → `recall()` receives it
2. Profiler looks up Feast online store for user profile + recent activity
3. Hybrid search (BM25 + vector + RRF) retrieves top-K episodic memories from Qdrant, filtered by `user_id`
4. Context assembler merges profile + memories into a single context string
5. Context string returned (no real LLM call — POC only)

---

## Architecture Decision 1: Chunking Strategy

**Decision:** Per-message chunking with max 512 tokens, no overlap.

**Alternatives considered:**
- **Per-conversation chunking:** Entire conversation as one chunk → retrieval quality drops because a single chunk contains multiple topics, making vector similarity less precise.
- **Semantic break:** Split on topic change (e.g., via embedding cosine distance threshold) → more accurate but O(n²) complexity, overkill for a POC. Also requires an extra embedding call per incoming message.

**Why per-message:** Each user message or document is a natural semantic unit. 512 tokens covers ~400 Vietnamese words (with code-switching overhead), which is enough for a typical message. No overlap keeps storage linear. The tradeoff is that very long documents (>512 tokens) are truncated — in production we'd add a secondary splitting layer based on markdown headers or sentence boundaries.

**Lab concept link:** This mirrors the `BATCH=64` batching pattern in NB1 — we embed in units, not all at once.

---

## Architecture Decision 2: Feature Schema

**Decision:** Tabular features (entity-based) rather than embedding features.

**Features defined:**

| Feature View | Entity | Fields | TTL | Source |
|-------------|--------|--------|:---:|--------|
| `user_profile_features` | user_id | reading_speed_wpm, preferred_language, topic_affinity | 30 days | Parquet batch |
| `activity_features` | user_id | queries_last_hour, distinct_topics_24h | 1 hour | Parquet stream |

**Alternatives considered:**
- **Embedding features:** Store a user's "interest vector" as a float32 array in Feast (e.g., 384-dim embedding of all their past queries). This would allow dot-product similarity at lookup time. However, it couples profile and vector search concerns, and updating the embedding on every new message requires re-indexing — expensive.

**Why tabular:** Simpler, clearer schema, matches `feature_views.py` from NB4 directly. The embedding is already handled by Qdrant (vector store). Feature store stays as structured, interpretable columns. The tradeoff: tabular features can't capture latent patterns (e.g., "user seems interested in cloud security even though their stated affinity is AI"). In production, a hybrid approach would use embedding features for latent signals and tabular for explicit ones.

**Lab concept link:** Directly mirrors NB4's 3 feature views with Entity → FeatureView → FileSource pattern, including TTL choices (30d for stable profile, 1h for activity).

---

## Architecture Decision 3: Freshness Strategy

**Decision:** Three-tier freshness per use case.

| Use Case | Freshness Need | Strategy | Mechanism |
|----------|:--------------:|----------|-----------|
| "Summarize what I read today" | Sub-second | Streaming push | After `remember()`, materialize user's feature view immediately |
| "Recommend what to read next" | 5-minute batch | Periodic materialize | `materialize-incremental` every 5 min via cron/background thread |
| "Weekly digest of my interests" | Daily | Daily batch | Full batch refresh at 2 AM |

**Alternatives considered:**
- **Sub-second for all:** Over-engineered — weekly digest doesn't need sub-second. Running streaming for everything increases infrastructure cost (Kafka/Redis memory) without user-facing benefit.
- **Daily for all:** Too slow for "what did I read today?" — user would get stale results.

**Why three-tier:** Matches the production pattern from Lab 19 §6 (Feast slide on streaming vs batch). The POC implements only the sub-second path (since `remember()` is called synchronously in the same process), but the architecture document acknowledges the other tiers for production.

**Lab concept link:** This implements the streaming vs batch tradeoff discussed in NB4's TTL section (30-day user profile vs 1-hour query velocity).

---

## Rejected Alternative

**Storing episodic memory as an embedding feature view in Feast.**

I considered this approach: encode each user message as a 384-dim embedding vector, store it as a `Float32` feature in a custom Feast feature view with `user_id` as entity, and use Feast's online store for retrieval. This would unify the storage layer (everything in Feast).

**Why rejected:**
1. **Re-index cycle mismatch:** New episodic memory arrives every few seconds (user chats); profile data changes daily/weekly. Running `feast materialize` every few seconds for memory is wasteful — Feast is optimized for batch/stream at minute granularity, not per-message.
2. **No vector search in Feast:** Feast's online store (SQLite/Redis) doesn't support approximate nearest neighbor search. Retrieving "top-5 most similar memories" would require scanning all rows — O(n) instead of O(log n) with Qdrant's HNSW index.
3. **Separation of concerns:** Vector store for search, feature store for structured features — each optimized for its own access pattern.

---

## Vietnamese-Context Considerations

### 1. Tokenization
The POC uses whitespace tokenization (`str.lower().split()`), which is the same as the main lab. For Vietnamese production:
- **Whitespace split** works poorly for Vietnamese because compound words ("tự động mở rộng") are split into separate tokens, losing semantic meaning.
- **Underthesea** (Vietnamese NLP library) provides proper word segmentation. However, it adds 5-10ms per query — acceptable for recall but not for indexing 1000+ docs.
- **pyvi** is lighter but less accurate.
- **Hybrid approach (recommended):** Use underthesea for query tokenization (since it's one query at a time), whitespace for batch indexing. This is the same "fast indexing, accurate query" pattern used in production systems.

### 2. Code-switching (Vi-En mix)
Vietnamese users frequently mix English technical terms (e.g., "cloud computing tiếng Việt"). The `bge-small-en-v1.5` model handles English well but underperforms on Vietnamese (as seen in NB2 where paraphrase Precision@10 was only 32% vs 96.7% for English-heavy exact queries). A multilingual model like `bge-m3` or `intfloat/multilingual-e5-large` would improve paraphrase recall by ~40% based on MTEB benchmarks.

### 3. Phonetic typos
Vietnamese phonetic typing (e.g., "t`u dong" instead of "tự động") breaks both BM25 and vector search. Solutions:
- **Pre-processing layer:** A character-level CNN or simple lookup table to normalize common phonetic patterns before embedding.
- **Fuzzy BM25:** Use edit distance fallback when exact BM25 returns < 3 results.

### 4. Privacy / Decree 13
Vietnamese data privacy regulations (Decree 13/2023/NĐ-CP) require user data localization and explicit consent for personalization. In production, this means:
- All memory data must be stored on Vietnamese-located servers.
- Users must have a "forget me" button that wipes both vector store and feature store data.
- PIT join audit trails (NB4 pattern) are essential for compliance audits.

---

## Honest Limitations (What This POC Doesn't Handle)

1. **No real LLM call** — the context string is returned but not fed to an actual language model
2. **No multi-user privacy isolation** — Qdrant uses a `user_id` payload filter, but there's no encryption per user
3. **No memory consolidation** — 5 similar memories remain as separate entries indefinitely
4. **No CRUD on memories** — `remember()` only appends; there's no update or delete
5. **No multi-device sync** — all data is in-process memory/SQLite, not shared across devices
6. **No authentication** — any caller can access any user's data
7. **vector dimension is hardcoded at 384** — tied to `bge-small-en-v1.5`; changing the model requires re-indexing the entire corpus

---

## Vibe Coding Workflow Log

Most effective prompt: *"Write a HybridMemoryAgent class that uses Qdrant for vector memory and Feast for user profile. The remember() method should embed text via fastembed and upsert to Qdrant with user_id filter. The recall() method should do hybrid search (BM25+vector+RRF) and assemble a context string with profile features from Feast."*

Prompt that failed: *"Make the agent automatically chunk long documents"* — the AI generated a generic text splitter without considering Vietnamese word boundaries. Had to manually specify: "Whitespace split for now; note underthesea for production."