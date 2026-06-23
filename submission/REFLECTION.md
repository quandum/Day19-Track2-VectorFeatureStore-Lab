# Reflection — Lab 19

**Tên:** Trần Mạnh Chánh Quân
**Mã học viên:** 2A202600786
**Cohort:** A20-K2
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên golden set 50 queries, BM25 (keyword) thắng ở exact queries (96.7%) nhờ khớp chính xác từ kỹ thuật. Vector (semantic) không thắng paraphrase queries (24%) do model `bge-small-en-v1.5` yếu trên tiếng Việt. Hybrid thắng trên mixed queries (100% vs 97-98%) và thắng overall (78.6% vs 77.8% keyword) nhờ kết hợp BM25 precision + vector recall qua RRF k=60.

Không dùng hybrid khi: (1) latency budget cực kỳ tight (hybrid ≈ keyword + semantic latency), (2) query set thuần exact term (BM25 đủ), (3) cost của 2 retriever > benefit, ví dụ single-region search với vocabulary hẹp.

---

## Điều ngạc nhiên nhất khi làm lab này

Ngạc nhiên nhất là BM25 trên exact queries (96.7%) outperformed vector (88.7%) dù vector là kỹ thuật hiện đại — điều này cho thấy simple lexical search vẫn rất mạnh cho technical queries, và hybrid mới là production default.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_