import os
import sys
import json
import argparse
import statistics
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.search import search_chunks_with_timing
from app.config import SIMILARITY_THRESHOLD, DB_BACKEND

QUESTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json")

def run_threshold_sweep(backend: str = None, embed_model: str = None):
    selected_backend = backend or os.getenv("DB_BACKEND", "postgres")
    target_embed = embed_model or os.getenv("EMBED_MODEL", "nomic-embed-text")
    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(f"Questions benchmark file not found at {QUESTIONS_FILE}")

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("=" * 80)
    print(f"SIMILARITY THRESHOLD SWEEP ({selected_backend.upper()} | {target_embed})")
    print(f"Live SIMILARITY_THRESHOLD (.env): {SIMILARITY_THRESHOLD}")
    print("=" * 80)

    answerable_items = []
    unanswerable_items = []

    for item in questions:
        cat = item.get("category", "")
        if item.get("expect_refusal"):
            unanswerable_items.append(item)
        elif cat in ("policy", "trap_condition", "trap_gap") or (cat == "mixed" and item.get("expected_sources")):
            answerable_items.append(item)

    print(f"Evaluating {len(answerable_items)} Answerable Policy/Trap/Mixed Questions vs {len(unanswerable_items)} Unanswerable Questions...\n")

    ans_scores = []
    print(f"{'ID':<6} | {'Cat':<15} | {'Score':<7} | {'Top Chunk Reference':<35} | Question")
    print("-" * 80)
    for it in answerable_items:
        q = it["q"]
        chunks, _, _ = search_chunks_with_timing(
            query=q,
            limit=1,
            retrieval_mode="vector",
            use_reranker=False,
            backend=selected_backend,
            embed_model=target_embed
        )
        score = round(chunks[0]["similarity"], 4) if chunks else 0.0
        ref = f"{chunks[0]['source_file']} (§{chunks[0]['section']})" if chunks else "N/A"
        ans_scores.append(score)
        print(f"{it['id']:<6} | {it['category']:<15} | {score:<7.4f} | {ref[:35]:<35} | {q[:35]}")

    unans_scores = []
    print("\n" + "-" * 80)
    for it in unanswerable_items:
        q = it["q"]
        chunks, _, _ = search_chunks_with_timing(
            query=q,
            limit=1,
            retrieval_mode="vector",
            use_reranker=False,
            backend=selected_backend,
            embed_model=target_embed
        )
        score = round(chunks[0]["similarity"], 4) if chunks else 0.0
        ref = f"{chunks[0]['source_file']} (§{chunks[0]['section']})" if chunks else "N/A"
        unans_scores.append(score)
        print(f"{it['id']:<6} | {it['category']:<15} | {score:<7.4f} | {ref[:35]:<35} | {q[:35]}")

    print("=" * 80)
    print("SCORE DISTRIBUTION SUMMARY:")
    print("-" * 80)
    ans_min = min(ans_scores) if ans_scores else 0.0
    ans_med = statistics.median(ans_scores) if ans_scores else 0.0
    ans_max = max(ans_scores) if ans_scores else 0.0

    unans_min = min(unans_scores) if unans_scores else 0.0
    unans_med = statistics.median(unans_scores) if unans_scores else 0.0
    unans_max = max(unans_scores) if unans_scores else 0.0

    print(f"  • Answerable (N={len(ans_scores):2d})   : Min = {ans_min:.4f} | Median = {ans_med:.4f} | Max = {ans_max:.4f}")
    print(f"  • Unanswerable (N={len(unans_scores):2d}) : Min = {unans_min:.4f} | Median = {unans_med:.4f} | Max = {unans_max:.4f}")

    print("\n" + "=" * 80)
    print(f"{'Threshold':<11} | {'Wrongly Refused (False -)':<26} | {'Wrongly Answered (False +)':<26} | Total Errors")
    print("-" * 80)

    best_thresh = 0.65
    min_errors = 9999
    thresholds = [round(0.30 + i * 0.05, 2) for i in range(13)]  # 0.30 to 0.90

    for t in thresholds:
        wrong_refused = sum(1 for s in ans_scores if s < t)
        wrong_answered = sum(1 for s in unans_scores if s >= t)
        total_err = wrong_refused + wrong_answered
        print(f"{t:<11.2f} | {wrong_refused:<26} | {wrong_answered:<26} | {total_err}")
        if total_err < min_errors:
            min_errors = total_err
            best_thresh = t

    print("-" * 80)
    print(f"\nRECOMMENDED SIMILARITY_THRESHOLD: {best_thresh:.2f}")
    print(f"Optimal Error Count at {best_thresh:.2f}: {min_errors}")
    print(f"Live Configuration in .env: SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep vector similarity threshold across eval questions")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--embed-model", type=str, default=None, help="Embedding model override (e.g. bge-m3, nomic-embed-text)")
    args = parser.parse_args()
    run_threshold_sweep(backend=args.backend, embed_model=args.embed_model)
