# Kế hoạch thực hiện Lab 19 — Vector Store + Feature Store (Track 2)

**Học viên:** Trần Mạnh Chánh Quân — **MSSV:** 2A202600786  
**Path:** Lite (default) — `fastembed` + Qdrant in-memory + SQLite Feast + FastAPI  
**Tổng điểm:** 100 pts core + 20 pts bonus (optional)

---

## Mục lục

1. [Giai đoạn 0 — Setup môi trường](#giai-đoạn-0--setup-môi-trường)
2. [Giai đoạn 1 — NB1: Embeddings & Vector Indexing (15 pts)](#giai-đoạn-1--nb1-embeddings--vector-indexing-15-pts)
3. [Giai đoạn 2 — NB2: Hybrid Search RRF (25 pts)](#giai-đoạn-2--nb2-hybrid-search-rrf-25-pts)
4. [Giai đoạn 3 — NB3: Search API Benchmark (25 pts)](#giai-đoạn-3--nb3-search-api-benchmark-25-pts)
5. [Giai đoạn 4 — NB4: Feast Feature Store (30 pts)](#giai-đoạn-4--nb4-feast-feature-store-30-pts)
6. [Giai đoạn 5 — Pipeline tổng & Kiểm tra (5 pts)](#giai-đoạn-5--pipeline-tổng--kiểm-tra-5-pts)
7. [Giai đoạn 6 — Hoàn thiện submission](#giai-đoạn-6--hoàn-thiện-submission)
8. [Giai đoạn 7 (Optional) — Bonus Challenge](#giai-đoạn-7-optional--bonus-challenge-20-bonus-pts)

---

## Giai đoạn 0 — Setup môi trường

| Bước | Hành động | Lệnh / Chi tiết | Kiểm tra |
|------|-----------|------------------|----------|
| 0.1 | Kiểm tra Python ≥ 3.10 | `python --version` | Phải ≥ 3.10 |
| 0.2 | Tạo venv + cài deps + seed corpus + smoke test | `bash setup-lite.sh` (~60s) | Script báo `All checks passed` |
| 0.3 | Verify lite path | `make verify-lite` | Exit code 0, không lỗi |
| 0.4 | Copy .env | Copy `.env.example` → `.env`, giữ `QDRANT_MODE=memory` | File `.env` tồn tại |
| 0.5 | Convert notebooks sang .ipynb | `make lab` (dừng sau khi convert) | File `.ipynb` xuất hiện trong `notebooks/` |

---

## Giai đoạn 1 — NB1: Embeddings & Vector Indexing (15 pts)

**File:** `notebooks/01_embeddings_index.ipynb` (source: `01_embeddings_index.py`)  
**Rubric mapping:**

| # | Criterion | Pts | Cách đạt |
|---|-----------|:---:|----------|
| 1 | `client.count("lab19").count == 1000` | 5 | Embed all 1000 docs và upsert vào Qdrant |
| 2 | Top-5 visible cho keyword query (cell §5) | 5 | Query "cloud computing và tự động mở rộng" → 5 docs hiển thị |
| 3 | Paraphrase query trả về top-5 dominated bởi `cloud` | 10 | Query "phương pháp tự động mở rộng hạ tầng theo lưu lượng người dùng" → không chứa "cloud" nhưng top-5 vẫn là cloud topic |

### Các bước thực hiện:

1. **Mở Jupyter Lab:** `make lab`, vào notebook `01_embeddings_index.ipynb`
2. **Chạy tuần tự các cells:**
   - Cell 1: Import + load corpus → `Corpus size: 1000 docs`
   - Cell 2: Load `fastembed` model → `Vector dim: 384`
   - Cell 3: Tạo Qdrant collection `lab19`
   - **Cell 4 (TODO):** Embed + upsert loop (batch=64):
     - Dùng `embedder.embed(texts)` với texts = `title + " " + text`
     - Tạo `PointStruct` với payload: `doc_id`, `topic`, `title`
     - Upsert vào Qdrant
     - Assert `client.count("lab19").count == 1000`
   - Cell 5: Query "cloud computing và tự động mở rộng" → top-5
   - Cell 6: Paraphrase query (không chứa "cloud") → top-5 thuộc `cloud`
3. **Chụp ảnh màn hình** (3 ảnh):
   - `nb1_indexed_1000.png` — output `Indexed: 1000 vectors`
   - `nb1_top5_keyword.png` — top-5 cho query keyword
   - `nb1_top5_paraphrase.png` — top-5 cho paraphrase query (chủ yếu cloud)

---

## Giai đoạn 2 — NB2: Hybrid Search RRF (25 pts)

**File:** `notebooks/02_hybrid_search_rrf.ipynb` (source: `02_hybrid_search_rrf.py`)  
**Rubric mapping:**

| # | Criterion | Pts | Cách đạt |
|---|-----------|:---:|----------|
| 4 | `search_hybrid` đúng RRF formula `1/(k + rank)`, rank 1-based | 10 | RRF code trong hàm `search_hybrid` |
| 5 | Avg Precision@10: hybrid > keyword AND hybrid > semantic | 10 | Bảng điểm tổng thể |
| 6 | Slice table: hybrid wins on `mixed`, vector on `paraphrase`, BM25 on `exact` | 5 | Bảng slice |

### Các bước thực hiện:

1. **Mở notebook** `02_hybrid_search_rrf.ipynb`
2. **Chạy tuần tự các cells:**
   - Cell 1: Import + reload corpus + build BM25 + Qdrant indices
   - Cell 2: Implement `search_keyword`, `search_semantic` (đã có sẵn)
   - **Cell 3 (TODO):** Implement `search_hybrid`:
     - Pull top-50 từ BM25 (`search_keyword(query, depth=50)`)
     - Pull top-50 từ vector (`search_semantic(query, depth=50)`)
     - RRF score: `1/(rrf_k + rank)` với `rank` 1-based
     - Sort descending, trả về top-10
     - **Check:** công thức `1/(k + rank)` chứ KHÔNG phải `1/rank`; rank bắt đầu từ 1, không phải 0
   - Cell 4: Precision@10 evaluation trên 50 golden queries
     - Assert hybrid avg > keyword avg AND hybrid avg > semantic avg
   - Cell 5: Slice table theo `mode_hint` (exact/paraphrase/mixed)
3. **Chụp ảnh màn hình** (2 ảnh):
   - `nb2_precision_table.png` — bảng Precision@10 tổng
   - `nb2_slice_table.png` — bảng slice theo query type

---

## Giai đoạn 3 — NB3: Search API Benchmark (25 pts)

**File:** `notebooks/03_search_api_benchmark.ipynb` (source: `03_search_api_benchmark.py`)  
**Rubric mapping:**

| # | Criterion | Pts | Cách đạt |
|---|-----------|:---:|----------|
| 7 | FastAPI `/search` trả về `SearchResponse` với `latency_ms` | 5 | Gọi API, verify response shape |
| 8 | P50/P95/P99 latency table cho 3 modes (server-side) | 10 | Benchmark 100 queries × 3 modes |
| 9 | Hybrid P99 server-side < 50 ms | 10 | Assert trong notebook |

### Các bước thực hiện:

1. **Mở notebook** `03_search_api_benchmark.ipynb`
2. **Chạy tuần tự các cells:**
   - Cell 1: Khởi động uvicorn subprocess, đợi healthz OK
   - Cell 2: Test single query hybrid → `latency_ms` hiển thị, top-3 hits
   - **Cell 3 (TODO):** Chạy benchmark:
     - 50 golden queries × 2 reps = 100 calls/mode
     - Đo `latency_ms` server-side (từ response body)
     - Tính P50/P95/P99
   - Cell 4: Assert hybrid P99 < 50ms → PASS
   - Cell 5: Dọn dẹp (terminate API server)
3. **Chụp ảnh màn hình** (3 ảnh):
   - `nb3_api_response.png` — 1 hybrid query response với top-3 hits
   - `nb3_latency_table.png` — bảng P50/P95/P99
   - `nb3_p99_pass.png` — hybrid P99 < 50ms PASS

---

## Giai đoạn 4 — NB4: Feast Feature Store (30 pts)

**File:** `notebooks/04_feast_feature_store.ipynb` (source: `04_feast_feature_store.py`)  
**Rubric mapping:**

| # | Criterion | Pts | Cách đạt |
|---|-----------|:---:|----------|
| 10 | `feast apply` succeeds → 3 feature views registered | 5 | Output `feast apply` hiển thị 3 feature views |
| 11 | `materialize-incremental` succeeds | 5 | Log hiển thị rows materialized |
| 12 | `get_online_features()` cho `user_id=u_001` valid | 5 | Dict với các features |
| 13 | 100-call online lookup P99 reported (< 10ms = full credit) | 5 | P99 < 10ms → PASS |
| 14 | PIT join via `get_historical_features()` → 3 rows × N features | 5 | DataFrame 3 rows |

### Các bước thực hiện:

1. **Mở notebook** `04_feast_feature_store.ipynb`
2. **Chạy tuần tự các cells:**
   - Cell 1: Import + tạo thư mục `app/feast_repo/data/`
   - Cell 2: Sinh 3 Parquet files:
     - `user_profile.parquet` (100 users)
     - `item_popularity.parquet` (1000 items)
     - `query_velocity.parquet` (100 users)
   - Cell 3: `feast apply` → 3 feature views registered
   - Cell 4: `feast materialize-incremental` → rows materialized
   - Cell 5: Online lookup cho `u_001` → valid dict + latency
   - **Cell 6 (TODO):** 100-call lookup benchmark → P50/P95/P99
   - Cell 7: PIT join với 3 users → DataFrame
3. **Chụp ảnh màn hình** (5 ảnh):
   - `nb4_parquet_files.png` — 3 file Parquet generated
   - `nb4_feast_apply.png` — STDOUT feast apply
   - `nb4_materialize.png` — log materialize
   - `nb4_online_lookup.png` — online lookup result + latency
   - `nb4_pit_join.png` — PIT join DataFrame

---

## Giai đoạn 5 — Pipeline tổng & Kiểm tra (5 pts)

| # | Criterion | Pts | Cách đạt |
|---|-----------|:---:|----------|
| 15 | Reproducible từ clean `bash setup-lite.sh && make benchmark` | 5 | Chạy lại từ đầu, `make benchmark` PASS với hybrid > kw & sem |

### Các bước thực hiện:

1. **Kiểm tra reproducibility:**
   - `make clean-lite` (xoá venv + data + Feast registry)
   - `bash setup-lite.sh` (fresh install)
   - `make benchmark` → hybrid beats both pure modes → exit 0
2. **Chạy pytest:**
   - `make test` → tất cả tests pass
3. **Kiểm tra output benchmark** có đúng:
   - Precision@10 hybrid > keyword AND hybrid > semantic
   - PASS line hiển thị

---

## Giai đoạn 6 — Hoàn thiện submission

### 6.1 Điền REFLECTION.md

**File:** `submission/REFLECTION.md`

Nội dung cần điền:
- **Tên:** Trần Mạnh Chánh Quân
- **Cohort:** A20 (cần xác nhận A20-K1 hay A20-K2)
- **Path đã chạy:** lite
- **Câu hỏi (≤200 chữ):** Trả lời về:
  - Mode nào thắng ở query nào? (exact → BM25, paraphrase → vector, mixed → hybrid)
  - Tại sao hybrid thắng overall? (kết hợp keyword precision + semantic recall)
  - Khi nào không dùng hybrid? (khi latency budget cực kỳ tight, khi chỉ có 1 loại query rõ ràng, hoặc khi cost của 2 retriever > benefit)
- **Điều ngạc nhiên nhất:** (optional, 1-2 câu)
- **Bonus challenge:** bỏ trống nếu không làm

### 6.2 Thêm ảnh chụp vào `submission/screenshots/`

Các ảnh cần có (theo rubric):

| File | Nội dung | NB |
|------|----------|:--:|
| `nb1_indexed_1000.png` | Indexed: 1000 vectors | 1 |
| `nb1_top5_keyword.png` | Top-5 cho query keyword | 1 |
| `nb1_top5_paraphrase.png` | Top-5 paraphrase query (cloud cluster) | 1 |
| `nb2_precision_table.png` | Bảng Precision@10 tổng thể | 2 |
| `nb2_slice_table.png` | Bảng slice theo query type | 2 |
| `nb3_api_response.png` | API response với top-3 hits | 3 |
| `nb3_latency_table.png` | Bảng P50/P95/P99 | 3 |
| `nb3_p99_pass.png` | Hybrid P99 < 50ms PASS | 3 |
| `nb4_parquet_files.png` | 3 Parquet files generated | 4 |
| `nb4_feast_apply.png` | feast apply STDOUT | 4 |
| `nb4_materialize.png` | materialize log | 4 |
| `nb4_online_lookup.png` | Online lookup result + latency | 4 |
| `nb4_pit_join.png` | PIT join DataFrame | 4 |

### 6.3 Commit & Push

```bash
git add -A
git commit -m "Lab 19 submission — Trần Mạnh Chánh Quân"
git push -u origin main
```

### 6.4 Nộp bài

Paste public GitHub URL vào ô submission Day 19 trong VinUni LMS.  
**Giữ repo public đến khi điểm được công bố.**

---

## Giai đoạn 7 (Optional) — Bonus Challenge (20 bonus pts)

**File tham khảo:** `BONUS-CHALLENGE.md`, `BONUS-CHALLENGE-EN.md`  
**Rubric:**

| Criterion | Pts | Cách đạt |
|-----------|:---:|----------|
| `bonus/ARCHITECTURE.md` exists, ≥ 600 words, architecture diagram | 3 | File markdown + hình |
| 3 architecture decisions với explicit tradeoff (X vs Y, why X) | 6 | Quyết định kiến trúc rõ ràng |
| Ít nhất 1 decision thể hiện Vietnamese-context awareness | 2 | Ví dụ: tiếng Việt tokenization, embedding model multilingual |
| Rejected alternative explicitly named with reason | 2 | Nêu rõ approach bị loại + lý do |
| `bonus/agent.py` runs (`HybridMemoryAgent.remember()` + `.recall()`) | 4 | Script chạy được |
| `bonus/demo.py` exits 0 với 5 query outputs printed | 3 | Demo chạy thành công |

### Các bước thực hiện (nếu làm bonus):

1. **Brainstorm (15 phút):** Thiết kế HybridMemoryAgent kết hợp Vector Store (episodic memory) + Feature Store (stable user profile)
2. **Tạo folder `bonus/`** với cấu trúc:
   ```
   bonus/
   ├── ARCHITECTURE.md
   ├── agent.py
   └── demo.py
   ```
3. **Viết ARCHITECTURE.md:** ≥ 600 words, vẽ architecture diagram, 3 decisions với tradeoff
4. **Implement agent.py:** HybridMemoryAgent với `remember()` và `recall()`
5. **Implement demo.py:** Chạy 5 queries, exit 0
6. **Push cùng với submission**

---

## Tổng hợp điểm dự kiến

| Giai đoạn | Notebook | Pts |
|:---------:|:--------:|:---:|
| 1 | NB1 — Embeddings & Vector Indexing | 15 |
| 2 | NB2 — Hybrid Search RRF | 25 |
| 3 | NB3 — Search API Benchmark | 25 |
| 4 | NB4 — Feast Feature Store | 30 |
| 5 | Pipeline tổng | 5 |
| 6 | Submission | — |
| | **Core total** | **100** |
| 7 (opt) | Bonus Challenge | 20 |

---

## Lưu ý quan trọng

1. **RRF rank 1-based:** Công thức `1/(k + rank)`, rank bắt đầu từ 1 (không phải 0) — sai điểm này mất 10 pts ở NB2.
2. **Server-side latency:** P99 < 50ms là server-side (từ `time.perf_counter()` trong `main.py`), không phải wall-clock.
3. **Reproducibility:** Trước khi nộp, chạy `make clean-lite && bash setup-lite.sh && make benchmark` để verify.
4. **Giữ repo public:** Nếu private, grader không xem được → 0 điểm.
5. **Output cells:** Giữ nguyên output cells trong `.ipynb` — grader sẽ kiểm tra.