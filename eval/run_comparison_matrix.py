#!/usr/bin/env python3
"""
eval/run_comparison_matrix.py
Automates Task 5 Configuration Comparison across:
- Config A: nomic, hybrid, separate gate (current default)
- Config C: bge-m3, vector, merged gate
- Config D: bge-m3, vector, separate gate
- Config E: bge-m3, hybrid, merged gate

Evaluates dev (40 questions) and held-out (22 questions) with judge OFF (--judge-model "").
Produces comparative summary table with Wilson CIs, retrieval metrics, stage latencies, and Ollama calls.
"""

import os
import sys
import json
import time
import math
import statistics
import subprocess
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIGS = [
    {
        "name": "Config A",
        "embed_model": "nomic-embed-text",
        "retrieval": "hybrid",
        "gate": "separate",
        "desc": "nomic, hybrid, sep gate (default)"
    },
    {
        "name": "Config C",
        "embed_model": "bge-m3",
        "retrieval": "vector",
        "gate": "merged",
        "desc": "bge-m3, vector, merged gate"
    },
    {
        "name": "Config D",
        "embed_model": "bge-m3",
        "retrieval": "vector",
        "gate": "separate",
        "desc": "bge-m3, vector, sep gate"
    },
    {
        "name": "Config E",
        "embed_model": "bge-m3",
        "retrieval": "hybrid",
        "gate": "merged",
        "desc": "bge-m3, hybrid, merged gate"
    }
]

def wilson_ci(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "n/a"
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    pct_val = 100 * p
    low = max(0.0, 100 * (center - spread))
    high = min(100.0, 100 * (center + spread))
    return f"{pct_val:.1f}% ({k}/{n}) [{low:.1f}%, {high:.1f}%]"

def run_single_eval(cfg: dict, split: str) -> dict:
    q_file = "eval/questions.json" if split == "dev" else "eval/questions_heldout.json"
    tag = f"comp_{cfg['name'].replace(' ', '_')}_{split}"
    extra_flags = f"--retrieval {cfg['retrieval']} --embed-model {cfg['embed_model']}"
    cmd = [
        sys.executable,
        "eval/run_eval.py",
        "--questions", q_file,
        "--gate", cfg["gate"],
        "--extra", extra_flags,
        "--judge-model", "",
        "--tag", tag
    ]
    print(f"\n>>> Running {cfg['name']} ({split}): {extra_flags} --gate {cfg['gate']}...")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"ERROR in {tag}:\n{p.stderr[-500:]}")
        
    # Find latest results file matching tag
    res_dir = ROOT / "eval" / "results"
    matching = sorted(res_dir.glob(f"*_{tag}.json"))
    if not matching:
        raise RuntimeError(f"No result file found for tag {tag}")
    latest_file = matching[-1]
    return json.loads(latest_file.read_text(encoding="utf-8"))

def analyze_results(results_data: dict) -> dict:
    rows = results_data.get("rows", [])
    n = len(rows)
    passed = sum(1 for r in rows if r.get("passed"))
    
    # Categories
    cats = {}
    for r in rows:
        c = r.get("category", "other")
        if c not in cats:
            cats[c] = {"passed": 0, "total": 0}
        cats[c]["total"] += 1
        if r.get("passed"):
            cats[c]["passed"] += 1
            
    # Retrieval metrics on answerable policy/mixed questions
    rel_rows = [r for r in rows if r.get("expected_sources") and not r.get("expect_refusal")]
    n_rel = len(rel_rows)
    hit1 = sum(1 for r in rel_rows if r.get("checks", {}).get("hit_1"))
    hit3 = sum(1 for r in rel_rows if r.get("checks", {}).get("hit_3"))
    hit5 = sum(1 for r in rel_rows if r.get("checks", {}).get("hit_5"))
    mrrs = [r.get("checks", {}).get("mrr", 0.0) for r in rel_rows if "mrr" in r.get("checks", {})]
    mean_mrr = statistics.mean(mrrs) if mrrs else 0.0
    
    # Citations, tools, refusal
    cit_ok = sum(1 for r in rel_rows if r.get("checks", {}).get("citation_ok"))
    
    tool_rows = [r for r in rows if "tools_ok" in r.get("checks", {})]
    tool_ok = sum(1 for r in tool_rows if r.get("checks", {}).get("tools_ok"))
    
    ref_rows = [r for r in rows if "refusal_ok" in r.get("checks", {})]
    ref_ok = sum(1 for r in ref_rows if r.get("checks", {}).get("refusal_ok"))
    
    # Latencies
    lats = [r.get("latency_s", 0.0) for r in rows]
    mean_lat = statistics.mean(lats) if lats else 0.0
    p95_lat = sorted(lats)[min(len(lats)-1, int(0.95 * len(lats)))] if lats else 0.0
    
    # Calls & Stages
    calls = [r.get("ollama_calls", 0) for r in rows if "ollama_calls" in r]
    mean_calls = statistics.mean(calls) if calls else 0.0
    
    stages = ["retrieval", "planner", "gate", "synthesis", "grounding_retry"]
    stage_means = {}
    for stg in stages:
        vals = [r.get("stage_timings", {}).get(stg, 0.0) for r in rows if r.get("stage_timings")]
        stage_means[stg] = statistics.mean(vals) if vals else 0.0
        
    return {
        "n": n,
        "passed": passed,
        "ci": wilson_ci(passed, n),
        "cats": {k: f"{v['passed']}/{v['total']}" for k, v in cats.items()},
        "n_rel": n_rel,
        "hit1": f"{100*hit1/n_rel:.1f}% ({hit1}/{n_rel})" if n_rel else "n/a",
        "hit3": f"{100*hit3/n_rel:.1f}% ({hit3}/{n_rel})" if n_rel else "n/a",
        "hit5": f"{100*hit5/n_rel:.1f}% ({hit5}/{n_rel})" if n_rel else "n/a",
        "mrr": f"{mean_mrr:.3f}",
        "citation_ok": f"{100*cit_ok/n_rel:.1f}% ({cit_ok}/{n_rel})" if n_rel else "n/a",
        "tool_ok": f"{100*tool_ok/len(tool_rows):.1f}% ({tool_ok}/{len(tool_rows)})" if tool_rows else "n/a",
        "refusal_ok": f"{100*ref_ok/len(ref_rows):.1f}% ({ref_ok}/{len(ref_rows)})" if ref_rows else "n/a",
        "mean_lat": round(mean_lat, 2),
        "p95_lat": round(p95_lat, 2),
        "mean_calls": round(mean_calls, 2),
        "stage_means": {k: round(v / 1000, 2) for k, v in stage_means.items()}
    }

def run_matrix():
    matrix = {}
    for cfg in CONFIGS:
        c_name = cfg["name"]
        matrix[c_name] = {"cfg": cfg}
        for split in ["dev", "heldout"]:
            res = run_single_eval(cfg, split)
            matrix[c_name][split] = analyze_results(res)
            
    # Save comparison matrix to json
    out_file = ROOT / "eval" / "comparison_matrix_results.json"
    out_file.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"\n[✓] Benchmark completed. Saved raw results to {out_file}\n")
    return matrix

if __name__ == "__main__":
    run_matrix()
