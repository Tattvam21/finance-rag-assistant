#!/usr/bin/env python3
"""
eval/gate_latency_benchmark.py
Benchmarks Answerability Gate latency and accuracy under:
- Current Baseline Gate: num_predict=150, full 5 chunks context (all retrieved policy text)
- Optimized Gate: think=False, num_predict=80, compact chunk context (top 3 chunks, trimmed)

Evaluates on:
1. 10 answerable policy questions (P01-P10)
2. 7 unanswerable questions (U01-U07)
Total = 17 test queries.
"""

import os
import sys
import json
import time
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_ollama_client, GATE_MODEL, NUM_CTX
from app.llm.prompts import ANSWERABILITY_GATE_PROMPT
from app.retrieval.search import search_chunks_with_timing
from app.utils.numeric_grounding import build_verified_numeric_facts

QUESTIONS_FILE = ROOT / "eval" / "questions.json"

def run_gate_variant(q_item: dict, variant: str, client, model: str):
    q_text = q_item["q"]
    expect_refusal = q_item.get("expect_refusal", False)
    
    # Retrieve top 5 chunks
    chunks, _, _ = search_chunks_with_timing(q_text, limit=5, retrieval_mode="hybrid", backend="postgres")
    
    if variant == "baseline":
        # Full 5 chunks
        context_blocks = [f"[{c['source_file']}, p.{c['page']}, §{c['section']}]:\n{c['content']}" for c in chunks]
        context_str = "\n\n".join(context_blocks)
        num_predict = 150
    elif variant == "optimized":
        # Top 3 chunks only + compact format
        context_blocks = [f"[{c['source_file']}, §{c['section']}]:\n{c['content'][:400]}" for c in chunks[:3]]
        context_str = "\n\n".join(context_blocks)
        num_predict = 80
    else:
        raise ValueError(f"Unknown variant: {variant}")
        
    num_facts, matching_quote = build_verified_numeric_facts(q_text, chunks, [])
    num_facts_block = f"\n{num_facts}\n" if num_facts else ""
    prompt = ANSWERABILITY_GATE_PROMPT.format(question=q_text, context=context_str, numeric_facts=num_facts_block)
    
    t0 = time.time()
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format={
            "type": "object",
            "properties": {
                "answerable": {"type": "boolean"},
                "supporting_quote": {"type": "string"}
            },
            "required": ["answerable", "supporting_quote"]
        },
        options={"temperature": 0.0, "num_ctx": NUM_CTX, "num_predict": num_predict},
        think=False,
        keep_alive="30m"
    )
    dur_ms = (time.time() - t0) * 1000
    
    prompt_tokens = getattr(resp, "prompt_eval_count", 0) or 0
    completion_tokens = getattr(resp, "eval_count", 0) or 0
    content = resp.message.content or "{}"
    try:
        data = json.loads(content)
        answerable = bool(data.get("answerable", False))
        quote = str(data.get("supporting_quote", "")).strip()
    except Exception:
        answerable = False
        quote = ""
        
    # Check correctness:
    # If expect_refusal is True -> correct if answerable is False
    # If expect_refusal is False -> correct if answerable is True
    is_correct = (not answerable) if expect_refusal else answerable
    
    return {
        "id": q_item["id"],
        "expect_refusal": expect_refusal,
        "answerable": answerable,
        "quote": quote,
        "is_correct": is_correct,
        "dur_ms": dur_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }

def run_gate_latency_benchmark():
    client = get_ollama_client()
    qs = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    
    test_ids = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10",
                "U01", "U02", "U03", "U04", "U05", "U06", "U07"]
    subset = [q for q in qs if q["id"] in test_ids]
    
    print("=" * 80)
    print(f"ANSWERABILITY GATE LATENCY & ACCURACY BENCHMARK ({GATE_MODEL})")
    print(f"Evaluating {len(subset)} questions (10 answerable, 7 unanswerable)")
    print("=" * 80)
    
    for variant in ["baseline", "optimized"]:
        print(f"\nRunning Variant: {variant.upper()}...")
        records = []
        for q in subset:
            rec = run_gate_variant(q, variant, client, GATE_MODEL)
            records.append(rec)
            status = "PASS" if rec["is_correct"] else "FAIL"
            print(f"  [{rec['id']}] {status} | {rec['dur_ms']:.1f}ms | InTokens: {rec['prompt_tokens']} | OutTokens: {rec['completion_tokens']} | Ans: {rec['answerable']}")
            
        acc = (sum(1 for r in records if r["is_correct"]) / len(records)) * 100
        mean_lat = statistics.mean(r["dur_ms"] for r in records)
        p95_lat = sorted(r["dur_ms"] for r in records)[int(0.95 * len(records))]
        mean_prompt_tok = statistics.mean(r["prompt_tokens"] for r in records)
        mean_out_tok = statistics.mean(r["completion_tokens"] for r in records)
        
        print("\n" + "-" * 80)
        print(f"SUMMARY FOR {variant.upper()}:")
        print(f"  • Gate Accuracy        : {acc:.1f}% ({sum(1 for r in records if r['is_correct'])}/{len(records)})")
        print(f"  • Mean Latency         : {mean_lat:.1f} ms ({mean_lat/1000:.2f} s)")
        print(f"  • P95 Latency          : {p95_lat:.1f} ms ({p95_lat/1000:.2f} s)")
        print(f"  • Mean Prompt Tokens   : {mean_prompt_tok:.0f} tokens")
        print(f"  • Mean Completion Tok  : {mean_out_tok:.0f} tokens")
        print("-" * 80)

if __name__ == "__main__":
    run_gate_latency_benchmark()
