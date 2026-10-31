#!/usr/bin/env python3
"""
eval/planner_stats.py
Calculates Query Planner statistics across dev and held-out test sets:
- Valid-JSON rate
- Fallback rate with categorized reasons
- Every question where the LLM plan differs from the keyword fallback plan
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CHAT_MODEL, get_ollama_client
from app.llm.generate import generate_query_plan, get_fallback_plan
from app.llm.schemas import QueryPlan

DEV_QUESTIONS = ROOT / "eval" / "questions.json"
HELDOUT_QUESTIONS = ROOT / "eval" / "questions_heldout.json"


def plan_to_dict(p: QueryPlan) -> Dict[str, Any]:
    d = p.model_dump()
    # Normalize empty fields
    if d.get("invoice"):
        # Drop None keys for comparison
        d["invoice"] = {k: v for k, v in d["invoice"].items() if v is not None}
    return d


def diff_plans(p_llm: QueryPlan, p_fb: QueryPlan) -> List[str]:
    diffs = []
    d1 = plan_to_dict(p_llm)
    d2 = plan_to_dict(p_fb)

    if d1["needs_policy"] != d2["needs_policy"]:
        diffs.append(f"needs_policy: LLM={d1['needs_policy']} vs FB={d2['needs_policy']}")

    if d1["late_fee_invoice_id"] != d2["late_fee_invoice_id"]:
        diffs.append(f"late_fee_invoice_id: LLM={d1['late_fee_invoice_id']} vs FB={d2['late_fee_invoice_id']}")

    inv1 = d1.get("invoice")
    inv2 = d2.get("invoice")

    if (inv1 is None) != (inv2 is None):
        diffs.append(f"invoice_filter: LLM={'present' if inv1 else 'None'} vs FB={'present' if inv2 else 'None'}")
    elif inv1 and inv2:
        for k in set(list(inv1.keys()) + list(inv2.keys())):
            v1 = inv1.get(k)
            v2 = inv2.get(k)
            if v1 != v2:
                diffs.append(f"invoice.{k}: LLM={v1} vs FB={v2}")

    return diffs


def run_planner_stats():
    client = get_ollama_client()
    dev_qs = json.loads(DEV_QUESTIONS.read_text(encoding="utf-8"))
    heldout_qs = json.loads(HELDOUT_QUESTIONS.read_text(encoding="utf-8"))

    all_items = [("dev", q) for q in dev_qs] + [("heldout", q) for q in heldout_qs]
    total = len(all_items)

    valid_json_count = 0
    fallback_count = 0
    fallback_reasons: Dict[str, int] = {}
    differing_questions = []

    print("=" * 80)
    print(f"QUERY PLANNER STATISTICS (Dev={len(dev_qs)}, Held-out={len(heldout_qs)}, Total={total})")
    print(f"Target Model: {CHAT_MODEL}")
    print("=" * 80)

    for idx, (split, q_item) in enumerate(all_items, 1):
        qid = q_item["id"]
        q_text = q_item["q"]

        # Run LLM planner
        plan_llm, source, dur = generate_query_plan(q_text, client, CHAT_MODEL, backend="postgres")

        # Run Fallback rule planner
        plan_fb = get_fallback_plan(q_text, backend="postgres")

        is_valid_json = source in ("planner", "fallback_augmented")
        if is_valid_json:
            valid_json_count += 1

        if source in ("fallback", "fallback_augmented"):
            fallback_count += 1
            reason = source
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

        # Check diffs
        diffs = diff_plans(plan_llm, plan_fb)
        if diffs:
            differing_questions.append({
                "id": qid,
                "split": split,
                "question": q_text,
                "source": source,
                "plan_llm": plan_to_dict(plan_llm),
                "plan_fb": plan_to_dict(plan_fb),
                "diffs": diffs
            })

        print(f"[{idx:02d}/{total}] {qid} ({split}): source={source} | diffs={len(diffs)}")

    valid_json_rate = (valid_json_count / total) * 100
    fallback_rate = (fallback_count / total) * 100

    print("\n" + "=" * 80)
    print("SUMMARY METRICS")
    print("-" * 80)
    print(f"Total Questions Evaluated : {total}")
    print(f"Valid JSON Rate           : {valid_json_count}/{total} ({valid_json_rate:.1f}%)")
    print(f"Fallback Rate             : {fallback_count}/{total} ({fallback_rate:.1f}%)")
    print("Fallback Reasons Breakdown:")
    for r, c in fallback_reasons.items():
        print(f"  • {r}: {c} questions")
    if not fallback_reasons:
        print("  • None (0 fallbacks triggered)")

    print("\n" + "=" * 80)
    print(f"QUESTIONS WHERE PLAN DIFFERS FROM KEYWORD FALLBACK ({len(differing_questions)}/{total})")
    print("-" * 80)
    for dq in differing_questions:
        print(f"\n[{dq['id']}] ({dq['split']}) \"{dq['question']}\"")
        print(f"  Source: {dq['source']}")
        print(f"  Differences:")
        for df in dq["diffs"]:
            print(f"    · {df}")
        print(f"  LLM Plan : {dq['plan_llm']}")
        print(f"  FB Plan  : {dq['plan_fb']}")


if __name__ == "__main__":
    run_planner_stats()
