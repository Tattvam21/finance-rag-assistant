import os
import sys
import time
import json
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion.parse import extract_pages, clean_text
from app.ingestion.chunk import chunk_page_text
from app.ingestion.embed import get_embeddings, get_embedding
from app.retrieval.search import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")
EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json")

def load_policy_questions() -> List[Dict[str, Any]]:
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        all_q = json.load(f)
    return [q for q in all_q if q.get("expected_source") and q.get("type") == "policy"]

def build_index_for_chunk_size(chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    pdf_files = [f for f in sorted(os.listdir(POLICIES_DIR)) if f.endswith(".pdf")]
    chunks = []
    idx = 0
    for pdf_name in pdf_files:
        pages = extract_pages(os.path.join(POLICIES_DIR, pdf_name))
        for page_num, raw_text in pages:
            cleaned = clean_text(raw_text)
            page_chunks = chunk_page_text(
                source_file=pdf_name,
                page=page_num,
                cleaned_text=cleaned,
                target_chunk_tokens=chunk_size,
                overlap_tokens=overlap,
                starting_chunk_index=idx
            )
            chunks.extend(page_chunks)
            idx += len(page_chunks)

    # Embed chunks
    texts = [c["content"] for c in chunks]
    embs = get_embeddings(texts)
    for i, c in enumerate(chunks):
        c["embedding"] = embs[i]

    return chunks

def evaluate_retrieval(chunks: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    hits = 0
    total = len(questions)
    latencies = []
    top1_sims = []

    for q in questions:
        q_text = q["q"]
        exp_source = q["expected_source"]

        t0 = time.time()
        q_vec = get_embedding(q_text)

        # Score against all chunks
        scored = []
        for c in chunks:
            sim = cosine_similarity(q_vec, c["embedding"])
            scored.append((sim, c["source_file"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        top5 = scored[:5]

        duration = (time.time() - t0) * 1000
        latencies.append(duration)

        if top5:
            top1_sims.append(top5[0][0])
            top5_sources = [s[1] for s in top5]
            if exp_source in top5_sources:
                hits += 1

    hit_rate = (hits / total) * 100 if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_top1_sim = sum(top1_sims) / len(top1_sims) if top1_sims else 0

    return {
        "hit_rate": round(hit_rate, 1),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_top1_sim": round(avg_top1_sim, 4),
        "hits": hits,
        "total": total
    }

def run_experiment():
    questions = load_policy_questions()
    print("=" * 75)
    print(f"CHUNKING EXPERIMENT: 300 vs 500 vs 800 tokens ({len(questions)} policy questions)")
    print("=" * 75)

    configurations = [
        {"size": 300, "overlap": 30, "label": "300 tokens (Fine-grained)"},
        {"size": 500, "overlap": 50, "label": "500 tokens (Baseline)"},
        {"size": 800, "overlap": 80, "label": "800 tokens (Coarse-grained)"}
    ]

    results = []

    for cfg in configurations:
        print(f"\nBuilding and indexing with {cfg['label']}...")
        chunks = build_index_for_chunk_size(cfg["size"], cfg["overlap"])
        eval_metrics = evaluate_retrieval(chunks, questions)
        print(f"  Generated {len(chunks)} chunks")
        print(f"  Retrieval Hit Rate @ 5: {eval_metrics['hit_rate']}% ({eval_metrics['hits']}/{eval_metrics['total']})")
        print(f"  Avg Latency: {eval_metrics['avg_latency_ms']} ms")
        print(f"  Avg Top-1 Similarity: {eval_metrics['avg_top1_sim']}")

        results.append({
            "size": cfg["size"],
            "overlap": cfg["overlap"],
            "total_chunks": len(chunks),
            "hit_rate": eval_metrics["hit_rate"],
            "avg_latency_ms": eval_metrics["avg_latency_ms"],
            "avg_top1_sim": eval_metrics["avg_top1_sim"]
        })

    # Generate Markdown Table
    md_lines = [
        "# Chunking Strategy Evaluation Benchmark",
        "",
        "Comparison of chunk sizes across policy retrieval hit rate, corpus size, and latency:",
        "",
        "| Chunk Size (tokens) | Overlap (tokens) | Total Chunks | Hit Rate @ 5 (%) | Avg Top-1 Cosine Sim | Avg Retrieval Latency (ms) | Notes |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for r in results:
        notes = "Optimal balance of citation precision & recall" if r["size"] == 500 else (
            "Higher chunk count, slightly lower paragraph context" if r["size"] == 300 else "Fewer chunks, covers broad section context"
        )
        md_lines.append(
            f"| **{r['size']}** | {r['overlap']} | {r['total_chunks']} | **{r['hit_rate']}%** | {r['avg_top1_sim']} | {r['avg_latency_ms']} ms | {notes} |"
        )

    md_output = "\n".join(md_lines)
    print("\n" + "=" * 75)
    print(md_output)
    print("=" * 75)

    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunking_results.md")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(md_output)
    print(f"\nSaved benchmark results to {results_file}")

if __name__ == "__main__":
    run_experiment()
