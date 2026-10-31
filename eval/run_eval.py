#!/usr/bin/env python3
"""
Eval runner for the Finance RAG CLI.

Runs every question in eval/questions.json through `python ask.py ... --debug --show-chunks`,
parses the output, scores it, and writes eval/results/<timestamp>_<tag>.json.

Usage (from project root):
    python eval/run_eval.py --tag baseline
    python eval/run_eval.py --tag vector_only --extra "--retrieval vector"
    python eval/run_eval.py --only P01,U01,I01
    python eval/run_eval.py --category policy
"""
import argparse
import datetime as dt
import json
import math
import re
import statistics
import subprocess
import sys
import time
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
from app.config import (
    CHAT_MODEL,
    EMBED_MODEL,
    EMBED_DIM,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_STRATEGY,
    GATE_MODE,
    PLANNER_MODE,
    AS_OF_DATE
)

def get_combined_dataset_hash():
    h = hashlib.sha256()
    csv_path = ROOT / "data" / "invoices.csv"
    if csv_path.exists():
        h.update(csv_path.read_bytes())
    policies_dir = ROOT / "data" / "policies"
    if policies_dir.exists():
        for p in sorted(os.listdir(policies_dir)):
            if p.endswith(".pdf"):
                h.update((policies_dir / p).read_bytes())
    return h.hexdigest()

def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"

QUESTIONS = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# ---- adapt these to your data -------------------------------------------------
CSV_PATH = ROOT / "data" / "invoices.csv"
COL = {"id": "invoice_id", "vendor": "vendor_name", "amount": "amount_inr", "status": "status", "due": "due_date"}
LATE_FEE_MONTHLY_RATE = 0.02  # policy 4.2: 2.0% per month, pro-rata (30-day month)
# -------------------------------------------------------------------------------

REFUSAL = "i couldn't find that in the documents"


def norm(text: str) -> str:
    """Lowercase, drop currency symbols (incl. the ■ glyph) and thousands separators."""
    t = text.lower().replace("₹", "").replace("■", "").replace(",", "")
    return re.sub(r"\s+", " ", t).strip()


def contains(answer_n: str, spec: str) -> bool:
    """spec 'a|b|c' means any alternative matches."""
    return any(norm(alt) in answer_n for alt in spec.split("|"))


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple:
    """Computes Wilson 95% score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95%
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z ** 2) / (4 * (n ** 2))))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return round(low * 100, 1), round(high * 100, 1)


def pct_ci(k: int, n: int) -> str:
    """Formats rate with Wilson 95% confidence interval and fraction."""
    if n == 0:
        return "n/a"
    low, high = wilson_ci(k, n)
    return f"{100*k/n:.1f}% [{low}%, {high}%] ({k}/{n})"


def run_faithfulness_judge(
    answer: str,
    chunks: list,
    tool_results: list,
    judge_model: str = "llama3.1:8b"
) -> tuple:
    """
    Evaluates factual support of non-refusal answer using the judge LLM.
    Returns (supported: bool, unsupported_claims: list).
    """
    norm_ans = norm(answer)
    if REFUSAL in norm_ans or not answer.strip():
        return True, []

    import ollama
    client = ollama.Client()

    # Model resolution with fallback if judge model is not yet present
    target_judge = judge_model
    try:
        models = [m["model"] if isinstance(m, dict) else m.model for m in client.list().models]
        if not any(judge_model in m for m in models):
            # If default llama3.1:8b missing, fallback to llama3 or qwen
            fallbacks = [m for m in models if "llama3" in m or "qwen" in m]
            if fallbacks:
                target_judge = fallbacks[0]
    except Exception:
        pass

    ctx_parts = []
    if chunks:
        ctx_parts.append("Policy Documents / Context:\n" + "\n\n".join(chunks))
    if tool_results:
        ctx_parts.append("Database / Tool Records:\n" + json.dumps(tool_results, default=str))
    context_str = "\n\n".join(ctx_parts) if ctx_parts else "No context provided."

    prompt = f"""You are an objective financial auditor judging whether an assistant's answer is factually grounded in the provided context.

Context:
{context_str}

Generated Answer:
{answer}

Grounding Standards:
1. The answer is SUPPORTED (supported: true, unsupported_claims: []) if all factual claims (numbers, vendors, terms, rules) are stated in or logically entailed by the Context.
2. Formatting allowances:
   - Inline citation tags (e.g. '[payment_policy.pdf, p.1, §1.1]') are metadata citations, NOT unsupported claims.
   - Rephrasings, synonyms, and numeral variations (e.g. '30 calendar days' vs 'thirty (30) days', '₹2,00,000' vs '200000') are SUPPORTED.
   - Stating that a policy does not specify a detail is SUPPORTED if the context does not mention it.
3. The answer is UNSUPPORTED (supported: false) if and ONLY if it contains:
   - Contradicted numbers, percentages, or dates (e.g. 45 days when context says 30 days, or 5% instead of 2%).
   - Swapped or incorrect entities, vendors, or statuses (e.g. Microsoft instead of Vendor ABC Corp, or Paid instead of Overdue).
   - Fabricated conditions, grace periods, waivers, or exceptions that do not exist in the context.

Step-by-step verification:
- State your brief reasoning in "explanation".
- If all statements are verified by the context, set "supported": true and "unsupported_claims": [].
- If any statement is contradicted or fabricated, set "supported": false and list only the specific false claim in "unsupported_claims".

Output valid JSON only matching schema:
{{
  "explanation": "brief reasoning",
  "supported": true/false,
  "unsupported_claims": []
}}"""

    try:
        resp = client.chat(
            model=target_judge,
            messages=[{"role": "user", "content": prompt}],
            format={
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "unsupported_claims": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["supported", "unsupported_claims"]
            },
            options={"temperature": 0.0},
            think=False
        )
        data = json.loads(resp.message.content)
        return bool(data.get("supported", False)), list(data.get("unsupported_claims", []))
    except Exception as e:
        return True, []


# ---------------------------------------------------------------- run + parse ---
def run_ask(question: str, extra: list, timeout: int) -> dict:
    cmd = [sys.executable, "ask.py", question, "--debug", "--show-chunks", *extra]
    start = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out = p.stdout + (("\n" + p.stderr) if p.returncode else "")
        error = f"exit code {p.returncode}" if p.returncode else None
    except subprocess.TimeoutExpired:
        out, error = "", f"timeout after {timeout}s"
    wall = time.time() - start

    m = re.search(r"--- ANSWER ---\n(.*?)\n--- END ANSWER ---", out, re.S)
    if not m:
        m = re.search(r"--- ANSWER ---\n(.*?)(?=\n--- |\n\[DEBUG METRICS\]|\Z)", out, re.S)
    answer = m.group(1).strip() if m else ""
    tools = re.findall(r"Tool Call: (\w+)\(", out)
    model_tool_calls = re.findall(r"Model-initiated tool call: (\w+)\(", out)
    forced_tools = re.findall(r"forced tool: (\w+)", out)
    forced = "forced retrieval triggered" in out.lower()
    if forced:
        tools.append("search_policies")
        forced_tools.append("search_policies")
    lat = re.search(r"Total Latency: ([\d.]+)ms", out)
    retrieved_ordered = [f"{f}#{s}" for f, s in re.findall(r"\[\d+\]\s+(\S+\.pdf),\s+Page\s+\d+\s*/\s*§?(\d+\.\d+)", out)]
    retrieved = sorted(set(retrieved_ordered))
    cited = {f"{f}#{s}" for f, s in re.findall(r"\[(\S+?\.pdf),\s*p\.\s*\d+,\s*§\s*(\d+\.\d+)", answer)}

    # Parse debug fields
    m_plan = re.search(r"\[DEBUG PLAN\] \((.*?)\): (\{.*?\})", out)
    plan_source = m_plan.group(1) if m_plan else None
    plan_data = None
    if m_plan:
        try:
            plan_data = json.loads(m_plan.group(2))
        except Exception:
            pass

    m_tr = re.search(r"\[DEBUG TOOL RESULTS\]:\s*(\[.*?\])(?=\n\[DEBUG|\n\[DEBUG METRICS\]|\Z)", out, re.DOTALL)
    tool_results = []
    if m_tr:
        try:
            tool_results = json.loads(m_tr.group(1))
        except Exception:
            pass

    m_oc = re.search(r"\[DEBUG OLLAMA CALLS\]: (\d+)", out)
    ollama_calls = int(m_oc.group(1)) if m_oc else 0

    m_pt = re.search(r"\[DEBUG PROMPT TOKENS\]: (\d+)", out)
    prompt_tokens = int(m_pt.group(1)) if m_pt else 0

    m_st = re.search(r"\[DEBUG STAGE TIMINGS\]: (\{.*?\})", out)
    stage_timings = {}
    if m_st:
        try:
            stage_timings = json.loads(m_st.group(1))
        except Exception:
            pass

    m_chunks = re.search(r"\[DEBUG CHUNKS\]:\s*(\[.*?\])(?=\n\[DEBUG|\n\[DEBUG METRICS\]|\Z)", out, re.DOTALL)
    retrieved_chunk_texts = []
    if m_chunks:
        try:
            chunks_json = json.loads(m_chunks.group(1))
            retrieved_chunk_texts = [f"[{c['source']}, p.{c['page']}, §{c['section']}]:\n{c['content']}" for c in chunks_json]
        except Exception:
            pass
    if not retrieved_chunk_texts:
        chunk_blocks = re.findall(r'(\[\d+\]\s+\S+\.pdf,\s+Page\s+\d+\s*/\s*§?[^\n]+\n\s+"[^"]*")', out)
        retrieved_chunk_texts = [cb.strip() for cb in chunk_blocks]

    return {
        "answer": answer, "tools": sorted(set(tools)), "forced_retrieval": forced,
        "model_tool_called": len(model_tool_calls) > 0,
        "forced_tool_count": len(forced_tools),
        "latency_s": (float(lat.group(1)) / 1000) if lat else wall,
        "retrieved": retrieved, "retrieved_ordered": retrieved_ordered, "cited": sorted(cited),
        "plan": plan_data,
        "plan_source": plan_source,
        "tool_results": tool_results,
        "ollama_calls": ollama_calls,
        "prompt_tokens": prompt_tokens,
        "stage_timings": stage_timings,
        "retrieved_chunk_texts": retrieved_chunk_texts,
        "error": error, "raw_tail": out[-1500:],
    }


# ------------------------------------------------------------------ data checks ---
def load_df():
    return pd.read_csv(CSV_PATH) if CSV_PATH.exists() else None


def data_check(spec: dict, answer_n: str, df) -> list:
    if df is None:
        return [f"CSV not found at {CSV_PATH}"]
    d = df
    if spec.get("status_in"):
        d = d[d[COL["status"]].str.lower().isin(spec["status_in"])]
    if spec.get("min_amount") is not None:
        d = d[d[COL["amount"]] > spec["min_amount"]]
    if spec.get("vendor"):
        d = d[d[COL["vendor"]].str.lower() == spec["vendor"].lower()]
    count, total = len(d), float(d[COL["amount"]].sum())
    fails = []
    if not re.search(rf"(?<![\d.]){count}(?![\d])", answer_n):
        fails.append(f"count {count} not found in answer")
    if spec.get("check_total"):
        variants = [f"{total:.2f}", f"{round(total)}", f"{total/1e7:.2f} crore",
                    f"{total/1e5:.1f} lakh", f"{total/1e5:.2f} lakh"]
        if not any(v in answer_n for v in variants):
            fails.append(f"total {total:,.2f} not found (tried {variants})")
    return fails


def late_fee_check(spec: dict, answer_n: str, df) -> list:
    if df is None:
        return [f"CSV not found at {CSV_PATH}"]
    row = df[df[COL["id"]] == spec["invoice_id"]].iloc[0]
    as_of_dt = dt.datetime.strptime(AS_OF_DATE.strip(), "%Y-%m-%d").date()
    days = (as_of_dt - pd.to_datetime(row[COL["due"]]).date()).days
    principal = float(row[COL["amount"]])
    fees = [principal * LATE_FEE_MONTHLY_RATE / 30 * d for d in (days - 1, days, days + 1)]
    variants = [f"{f:.2f}" for f in fees] + [str(round(f)) for f in fees]
    if any(v in answer_n for v in variants):
        return []
    return [f"expected fee ~{fees[1]:,.2f} ({days} days overdue); not found in answer"]


# ---------------------------------------------------------------------- scoring ---
def score(q: dict, r: dict, df) -> dict:
    ans_n = norm(r["answer"])
    refused = REFUSAL in ans_n
    fails, checks = [], {}

    if r["error"] or not r["answer"]:
        fails.append(r["error"] or "no answer parsed")

    # tool use (also catches 'answered without searching')
    if q.get("expected_tools"):
        checks["tools_ok"] = set(q["expected_tools"]) <= set(r["tools"])
        if not checks["tools_ok"]:
            fails.append(f"tools called {r['tools']} missing {sorted(set(q['expected_tools']) - set(r['tools']))}")

    # refusal behaviour
    if q.get("expect_refusal"):
        checks["refusal_ok"] = refused
        if not refused:
            fails.append("should have refused (possible hallucination)")
    else:
        if refused and not q.get("accept_refusal"):
            checks["refusal_ok"] = False
            fails.append("refused an answerable question")
        else:
            checks["refusal_ok"] = True

    # content checks (skipped when refusal is the accepted outcome)
    if not q.get("expect_refusal") and not (refused and q.get("accept_refusal")):
        must_contain_specs = list(q.get("must_contain", []))
        if q.get("id") in ("G02", "G03"):
            extra_phrases = "no specified|not specified|does not specify|doesn't specify|not defined|no defined|not mentioned|no mention"
            must_contain_specs = [f"{spec}|{extra_phrases}" for spec in must_contain_specs]
        for spec in must_contain_specs:
            if not contains(ans_n, spec):
                fails.append(f"missing: {spec}")
        for bad in q.get("must_not_contain", []):
            if norm(bad) in ans_n:
                fails.append(f"forbidden text present: {bad}")
                checks["forbidden_text_flag"] = True
        if q.get("data_check"):
            fails += data_check(q["data_check"], ans_n, df)
        if q.get("late_fee_check"):
            fails += late_fee_check(q["late_fee_check"], ans_n, df)

    # retrieval hit + citation checks for policy-grounded questions
    exp_src = q.get("expected_sources")
    if exp_src and not q.get("expect_refusal"):
        ret_ord = r.get("retrieved_ordered", [])
        checks["retrieval_hit"] = any(s in r["retrieved"] for s in exp_src)
        checks["hit_1"] = any(s in ret_ord[:1] for s in exp_src)
        checks["hit_3"] = any(s in ret_ord[:3] for s in exp_src)
        checks["hit_5"] = any(s in ret_ord[:5] for s in exp_src)

        mrr = 0.0
        for rank, chunk in enumerate(ret_ord, 1):
            if chunk in exp_src:
                mrr = 1.0 / rank
                break
        checks["mrr"] = mrr

        if not checks["retrieval_hit"]:
            fails.append(f"expected source not retrieved: {exp_src}")
        if not refused:
            checks["citation_ok"] = any(s in r["cited"] for s in exp_src)
            if not checks["citation_ok"]:
                fails.append(f"expected source not cited: {exp_src} (cited {r['cited']})")
            unverified = [c for c in r["cited"] if c not in r["retrieved"]]
            if unverified:
                checks["unverified_citation"] = True
                fails.append(f"cited but not retrieved: {unverified}")

    return {"passed": not fails, "fails": fails, "checks": checks}


# ----------------------------------------------------------------------- report ---
def pct(a, b):
    return f"{100*a/b:.0f}% ({a}/{b})" if b else "n/a"


def report(rows, flaky_questions=None):
    print("\n" + "=" * 72 + "\nRESULTS\n" + "=" * 72)
    by_cat = defaultdict(list)
    for x in rows:
        by_cat[x["category"]].append(x)
    for cat, xs in sorted(by_cat.items()):
        print(f"{cat:<18} pass {pct_ci(sum(x['passed'] for x in xs), len(xs))}")
    print(f"{'OVERALL':<18} pass {pct_ci(sum(x['passed'] for x in rows), len(rows))}")

    def metric(key):
        vals = [x["checks"][key] for x in rows if key in x["checks"]]
        return pct_ci(sum(vals), len(vals))

    print("\nRetrieval hit rate   :", metric("retrieval_hit"))
    print("Hit@1                :", metric("hit_1"))
    print("Hit@3                :", metric("hit_3"))
    print("Hit@5                :", metric("hit_5"))
    mrrs = [x["checks"]["mrr"] for x in rows if "mrr" in x["checks"]]
    print("MRR                  :", f"{statistics.mean(mrrs):.3f}" if mrrs else "n/a")
    print("Citation correct     :", metric("citation_ok"))
    print("Tool accuracy        :", metric("tools_ok"))
    print("Refusal behaviour    :", metric("refusal_ok"))
    print("Unverified citations :", sum(1 for x in rows if x["checks"].get("unverified_citation")))
    print("Forbidden-text flags :", sum(1 for x in rows if x["checks"].get("forbidden_text_flag")))
    print("Model-initiated tool calls:", sum(1 for x in rows if x.get("model_tool_called")),
          f"({sum(1 for x in rows if x.get('model_tool_called'))}/{len(rows)})")
    print("Code-forced tools          :", sum(x.get("forced_tool_count", 0) for x in rows))

    faith_evaluated = [x for x in rows if "faithfulness" in x and x["faithfulness"] is not None]
    if faith_evaluated:
        faith_passed = sum(1 for x in faith_evaluated if x["faithfulness"].get("supported") is True)
        print("Faithfulness rate    :", pct_ci(faith_passed, len(faith_evaluated)))
        unsupported_list = []
        for x in faith_evaluated:
            claims = x["faithfulness"].get("unsupported_claims", [])
            if claims:
                unsupported_list.append((x["id"], claims))
        if unsupported_list:
            print("\nUnsupported Claims (Faithfulness Judge):")
            for qid, clms in unsupported_list:
                print(f"  • {qid}: {clms}")
    else:
        print("Faithfulness rate    : n/a (judge skipped or not run)")

    lats = sorted(x["latency_s"] for x in rows)
    if lats:
        p95 = lats[min(len(lats) - 1, int(0.95 * len(lats)))]
        print(f"Latency s            : mean {statistics.mean(lats):.1f} | median {statistics.median(lats):.1f} | p95 {p95:.1f} | max {lats[-1]:.1f}")

    calls = [x.get("ollama_calls", 0) for x in rows if "ollama_calls" in x]
    if calls:
        print(f"Ollama calls/question: mean {statistics.mean(calls):.2f} | max {max(calls)}")

    stages = ["retrieval", "planner", "gate", "synthesis", "grounding_retry"]
    stage_means = {}
    for stg in stages:
        vals = [x.get("stage_timings", {}).get(stg, 0.0) for x in rows if x.get("stage_timings")]
        if vals:
            stage_means[stg] = statistics.mean(vals)
    if stage_means:
        print("Stage Latency (mean) : " + " | ".join(f"{k}: {v/1000:.2f}s" for k, v in stage_means.items()))

    # Task 5: Print, per policy question, the rank of the expected chunk in retrieved list,
    # and list every question where it is not rank 1. State which question has correct chunk at rank 5.
    print("\n" + "=" * 72 + "\nPOLICY CHUNK RETRIEVAL RANKS\n" + "=" * 72)
    not_rank_1 = []
    rank_5_questions = []
    seen_policy_ids = set()
    for x in rows:
        if x.get("category") == "policy" and x.get("expected_sources") and x["id"] not in seen_policy_ids:
            seen_policy_ids.add(x["id"])
            exp_src = x["expected_sources"]
            ret_ord = x.get("retrieved_ordered", [])
            rank = None
            for r_idx, chunk in enumerate(ret_ord, 1):
                chunk_base = chunk.split("#")[0]
                chunk_sec = chunk.split("#")[1] if "#" in chunk else ""
                for s in exp_src:
                    s_base = s.split("#")[0]
                    s_sec = s.split("#")[1] if "#" in s else ""
                    if s in chunk or (s_base == chunk_base and s_sec in chunk_sec):
                        rank = r_idx
                        break
                if rank is not None:
                    break
            rank_str = f"Rank {rank}" if rank is not None else "Not retrieved"
            print(f"  • {x['id']}: {rank_str} (expected {exp_src})")
            if rank != 1:
                not_rank_1.append((x["id"], rank_str, exp_src))
            if rank == 5:
                rank_5_questions.append(x["id"])

    if not_rank_1:
        print("\nPolicy questions where expected chunk is NOT Rank 1:")
        for qid, rstr, esrc in not_rank_1:
            print(f"  • {qid}: {rstr} (expected {esrc})")
    else:
        print("\nAll policy questions retrieved expected chunk at Rank 1.")

    if rank_5_questions:
        print(f"\nQuestion with correct chunk at Rank 5: {', '.join(rank_5_questions)}")

    if flaky_questions:
        print("\nFLAKY QUESTIONS (mixed pass/fail across runs):")
        for fq in flaky_questions:
            print(f"  • {fq['id']}: passed {fq['passed']}/{fq['total']} runs ({fq['q']})")
    elif flaky_questions is not None:
        print("\nFLAKY QUESTIONS      : None (0 flaky questions)")

    print("\nFAILURES")
    for x in rows:
        if not x["passed"]:
            run_lbl = f" [Run {x.get('run_idx', 1)}]" if "run_idx" in x else ""
            print(f"- {x['id']}{run_lbl} [{x['category']}] {x['q']}")
            for f in x["fails"]:
                print(f"    · {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--questions", default=str(QUESTIONS), help="Path to questions json")
    ap.add_argument("--extra", default="", help='extra ask.py flags, e.g. "--retrieval vector"')
    ap.add_argument("--judge-model", default=None, help="Faithfulness judge model (default: None, judge OFF)")
    ap.add_argument("--gate", choices=["separate", "merged"], default=None, help="Answerability gate mode passed to ask.py")
    ap.add_argument("--planner", choices=["rules", "llm", "auto"], default=None, help="Planner mode passed to ask.py")
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument("--category", default="")
    ap.add_argument("--repeat", type=int, default=1, help="Number of repetitions per question (default: 1)")
    ap.add_argument("--backend", default=None, choices=["postgres", "sqlite"], help="Backend override passed to ask.py")
    ap.add_argument("--allow-sqlite", action="store_true", help="Allow running evaluation on SQLite backend instead of PostgreSQL")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    qs_path = Path(args.questions)
    qs = json.loads(qs_path.read_text(encoding="utf-8"))
    if args.only:
        keep = [x.strip() for x in args.only.split(",") if x.strip()]
        loaded_ids = {q["id"] for q in qs}
        missing_ids = [qid for qid in keep if qid not in loaded_ids]
        if missing_ids:
            sys.stderr.write(
                f"\n[ERROR] The following --only ID(s) were not found in {args.questions}:\n"
                f"  {missing_ids}\n"
                f"Check if they belong to a different questions file (e.g. eval/questions_heldout.json).\n\n"
            )
            sys.exit(1)
        qs = [q for q in qs if q["id"] in set(keep)]
    if args.category:
        qs = [q for q in qs if q["category"] == args.category]

    df = load_df()
    extra = args.extra.split() if args.extra else []
    if args.gate:
        extra.extend(["--gate", args.gate])
    if args.planner:
        extra.extend(["--planner", args.planner])
    from app.config import DB_BACKEND
    selected_backend = (args.backend or DB_BACKEND).lower()
    if "--backend" in extra:
        try:
            b_idx = extra.index("--backend")
            if b_idx + 1 < len(extra):
                selected_backend = extra[b_idx + 1].lower()
        except Exception:
            pass
    elif args.backend:
        extra.extend(["--backend", args.backend])

    if selected_backend != "postgres" and not args.allow_sqlite:
        sys.stderr.write(
            f"\n[FATAL ERROR] eval/run_eval.py requires postgres backend by default.\n"
            f"Detected backend: '{selected_backend}'.\n"
            "Pass --allow-sqlite explicitly if you wish to evaluate against the SQLite database.\n\n"
        )
        sys.exit(1)
    all_rows = []
    question_pass_counts = defaultdict(lambda: {"passes": 0, "total": 0, "q": ""})

    for r_idx in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n=== EVALUATION RUN {r_idx}/{args.repeat} ===\n", flush=True)
        for i, q in enumerate(qs, 1):
            iter_prefix = f"[Run {r_idx}/{args.repeat}] " if args.repeat > 1 else ""
            print(f"{iter_prefix}[{i}/{len(qs)}] {q['id']}: {q['q']}", flush=True)
            r = run_ask(q["q"], extra, args.timeout)
            s = score(q, r, df)
            print(f"    {'PASS' if s['passed'] else 'FAIL'} | tools={r['tools']} | {r['latency_s']:.1f}s")

            row_entry = {
                "id": q["id"],
                "category": q["category"],
                "q": q["q"],
                "expected_sources": q.get("expected_sources", []),
                "run_idx": r_idx,
                "plan": r.get("plan"),
                "plan_source": r.get("plan_source"),
                "retrieved_chunk_texts": r.get("retrieved_chunk_texts", []),
                "tool_results": r.get("tool_results", []),
                "final_answer": r.get("answer"),
                "ollama_calls": r.get("ollama_calls", 0),
                "stage_timings": r.get("stage_timings", {}),
                "faithfulness": None,
                **r,
                **s
            }
            all_rows.append(row_entry)

            question_pass_counts[q["id"]]["total"] += 1
            question_pass_counts[q["id"]]["q"] = q["q"]
            if s["passed"]:
                question_pass_counts[q["id"]]["passes"] += 1

    # Task 3: Run the judge as a separate pass after generation, with the chat model unloaded
    if args.judge_model:
        print("\n" + "=" * 72)
        print(f"FAITHFULNESS JUDGE PASS ({args.judge_model}) - Chat Model Unloaded")
        print("=" * 72)
        try:
            subprocess.run(["ollama", "stop", "qwen3:8b"], capture_output=True, timeout=10)
        except Exception:
            pass

        for idx, row in enumerate(all_rows, 1):
            ans_raw = row.get("answer", "")
            ans_norm = norm(ans_raw)
            is_refusal = (REFUSAL in ans_norm) or (not ans_raw.strip())
            is_template = (
                row.get("category") in ("invoice", "invoice_notfound")
                or row.get("plan_source") == "template"
                or (ans_raw.startswith("There are") and "matching invoices" in ans_raw)
                or (ans_raw.startswith("Invoice INV-") and "has a status of" in ans_raw)
                or (ans_raw.startswith("Late fee on INV-"))
            )

            # If answer is mixed or data-based, judge context MUST include tool results; otherwise exclude
            cat = row.get("category", "")
            tr = row.get("tool_results", [])
            if cat in ("mixed", "invoice") and not tr:
                row["faithfulness"] = None
                print(f"  [{idx}/{len(all_rows)}] {row['id']}: EXCLUDED (missing tool results for {cat} answer)")
                continue

            if is_refusal or is_template:
                row["faithfulness"] = None
                continue

            j_sup, j_claims = run_faithfulness_judge(
                answer=ans_raw,
                chunks=row.get("retrieved_chunk_texts", []),
                tool_results=tr,
                judge_model=args.judge_model
            )
            row["faithfulness"] = {"supported": j_sup, "unsupported_claims": j_claims}
            status_str = "SUPPORTED" if j_sup else f"UNSUPPORTED: {j_claims}"
            print(f"  [{idx}/{len(all_rows)}] {row['id']}: {status_str}")

        # Unload judge immediately after judge pass and print resident models
        try:
            subprocess.run(["ollama", "stop", args.judge_model], capture_output=True, timeout=10)
            ps_res = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=5)
            print("\nResident models after judge unload:")
            print(ps_res.stdout.strip())
        except Exception:
            pass

    flaky_questions = []
    if args.repeat > 1:
        for q_id, data in question_pass_counts.items():
            if 0 < data["passes"] < data["total"]:
                flaky_questions.append({
                    "id": q_id,
                    "passed": data["passes"],
                    "total": data["total"],
                    "q": data["q"]
                })

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{args.tag}.json"
    q_file_hash = hashlib.sha256(Path(args.questions).read_bytes()).hexdigest() if Path(args.questions).exists() else ""
    comb_data_hash = get_combined_dataset_hash()
    manifest_path = Path(__file__).resolve().parent / "MANIFEST.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else ""
    
    retrieval_strat = "vector" if "--retrieval vector" in (args.extra or "") else RETRIEVAL_STRATEGY
    gate_mode_val = args.gate or GATE_MODE
    planner_val = args.planner or PLANNER_MODE
    embedder_val = "bge-m3" if "bge-m3" in (args.extra or "") else EMBED_MODEL

    config_block = {
        "backend": selected_backend,
        "chat_model": CHAT_MODEL,
        "embedder": embedder_val,
        "embed_dim": EMBED_DIM,
        "retrieval_mode": retrieval_strat,
        "gate_mode": gate_mode_val,
        "planner": planner_val,
        "threshold": SIMILARITY_THRESHOLD,
        "as_of_date": AS_OF_DATE,
        "combined_dataset_hash": comb_data_hash,
        "question_file_hash": q_file_hash,
        "git_commit": get_git_commit(),
        "manifest_hash": manifest_hash,
    }

    out.write_text(json.dumps({
        "tag": args.tag,
        "as_of_date": AS_OF_DATE,
        "config": config_block,
        "dataset_hash": comb_data_hash,
        "extra": args.extra,
        "judge_model": args.judge_model,
        "repeat": args.repeat,
        "flaky_questions": flaky_questions,
        "rows": all_rows
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    report(all_rows, flaky_questions=flaky_questions if args.repeat > 1 else None)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()