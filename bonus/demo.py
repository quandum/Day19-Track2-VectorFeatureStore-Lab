#!/usr/bin/env python
"""Demo script for HybridMemoryAgent — 5 queries showcasing episodic memory + profile.

Usage:
    cd <repo-root>
    python bonus/demo.py

Ensure:
  - Feast has been applied + materialized (NB4 was run)
  - .venv is active
"""
import sys
import time
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bonus.agent import HybridMemoryAgent


def main() -> int:
    print("=" * 60)
    print("  HybridMemoryAgent — 5-Query Demo")
    print("  Học viên: Trần Mạnh Chánh Quân")
    print("=" * 60)

    # ── Seed some episodic memories ──────────────────────────
    agent = HybridMemoryAgent()
    memories = [
        ("u_001", "Kubernetes helps auto-scale containerized applications based on CPU load"),
        ("u_001", "Cloud security best practices include encryption at rest and in transit"),
        ("u_001", "I read a paper about transformer attention mechanisms for NLP"),
        ("u_001", "Vector databases like Qdrant are optimized for similarity search at scale"),
        ("u_001", "Feature stores like Feast provide a single source of truth for ML features"),
        ("u_001", "The Vietnamese AI market is growing fast with many startups in Ho Chi Minh City"),
        ("u_001", "Auto-scaling infrastructure can reduce cloud costs by up to 40%"),
    ]
    print("\n📝 Seeding episodic memories...")
    for uid, text in memories:
        agent.remember(text, user_id=uid)
        print(f"  [{uid}] remembered: {text[:60]}...")

    # ── 5 Demo Queries ───────────────────────────────────────
    queries = [
        {
            "q": "What have I read about Kubernetes?",
            "desc": "Simple lookup (vector hit expected)",
            "user_id": "u_001",
        },
        {
            "q": "Recommend what to read next",
            "desc": "Profile context needed (topic_affinity)",
            "user_id": "u_001",
        },
        {
            "q": "What am I focused on lately?",
            "desc": "Fresh activity check (queries_last_hour)",
            "user_id": "u_001",
        },
        {
            "q": "Tài liệu về tự động mở rộng hạ tầng",
            "desc": "Paraphrase query (vector wins over BM25)",
            "user_id": "u_001",
        },
        {
            "q": "Give me a cloud security summary",
            "desc": "Mixed query — hybrid search + profile",
            "user_id": "u_001",
        },
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n{'─' * 60}")
        print(f"📌 Query {i}: {q['desc']}")
        print(f"   \"{q['q']}\"")
        print(f"{'─' * 60}")
        t0 = time.perf_counter()
        context = agent.recall(q["q"], user_id=q["user_id"])
        elapsed = (time.perf_counter() - t0) * 1000
        print(context)
        print(f"  (assembled in {elapsed:.1f}ms)")

    print("\n" + "=" * 60)
    print("  ✅ Demo complete — all 5 queries ran successfully")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())