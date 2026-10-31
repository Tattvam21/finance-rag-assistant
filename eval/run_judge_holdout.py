#!/usr/bin/env python3
"""
eval/run_judge_holdout.py
Runs the calibrated faithfulness judge (llama3.1:8b) ONCE on eval/judge_holdout.json
with chat model (qwen3:8b) unloaded.
Reports per-item verdicts, judge evidence, correct-marked-supported %, and flawed-caught %.
"""

import sys
import json
import re
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ollama
from app.db import get_connection

PROMPT_V3_CALIBRATED = """You are an objective financial auditor judging whether an assistant's answer is factually grounded in the provided context.

Context:
{context}

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
  "supported": true,
  "unsupported_claims": []
}}"""

def unload_chat_model():
    print("[1/3] Unloading chat model (qwen3:8b)...")
    try:
        subprocess.run(["ollama", "stop", "qwen3:8b"], capture_output=True, timeout=10)
    except Exception as e:
        print(f"Warning unloading qwen3:8b: {e}")

def main():
    unload_chat_model()

    print("[2/3] Loading chunks and holdout dataset...")
    conn, _ = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT source_file, section, content FROM chunks_bge")
    all_chunks = cur.fetchall()

    holdout_file = ROOT / "eval" / "judge_holdout.json"
    items = json.loads(holdout_file.read_text(encoding="utf-8"))

    client = ollama.Client()
    judge_model = "llama3.1:8b"

    print(f"[3/3] Running calibrated judge on {len(items)} holdout items...")

    correct_total = 0
    correct_marked_supported = 0
    flawed_total = 0
    flawed_caught = 0

    results = []
    lines = []
    lines.append("=" * 90)
    lines.append(f"FAITHFULNESS JUDGE HOLDOUT RUN — MODEL: {judge_model}")
    lines.append(f"Target: correct_marked_supported >= 90%, flawed_caught >= 90%")
    lines.append("=" * 90)

    for idx, item in enumerate(items, 1):
        iid = item["id"]
        q = item["question"]
        ans = item["answer"]
        exp_sup = item["expected_supported"]
        cat = item["category"]
        flaw_desc = item.get("flaw_description", "N/A")

        # Map citation in answer to chunk
        m = re.search(r'\[([a-z_]+\.pdf).*?§([0-9\.]+)', ans)
        if not m:
            raise ValueError(f"Could not parse citation in answer for {iid}: {ans}")
        doc, sec = m.group(1), m.group(2)
        matched = [c for c in all_chunks if c[0] == doc and c[1].startswith(sec)]
        if not matched:
            raise ValueError(f"No chunk matched for {iid}: {doc} §{sec}")
        
        chunk_name = f"{matched[0][0]} §{matched[0][1]}"
        chunk_content = matched[0][2]

        prompt = PROMPT_V3_CALIBRATED.format(
            context=f"[{chunk_name}]:\n{chunk_content}",
            answer=ans
        )

        t0 = time.time()
        resp = client.chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            format={
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "unsupported_claims": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["explanation", "supported", "unsupported_claims"]
            },
            options={"temperature": 0.0},
            think=False
        )
        latency = round(time.time() - t0, 2)

        data = json.loads(resp.message.content)
        act_sup = bool(data.get("supported", False))
        explanation = data.get("explanation", "")
        claims = data.get("unsupported_claims", [])

        if exp_sup:
            correct_total += 1
            is_pass = (act_sup == True)
            if is_pass:
                correct_marked_supported += 1
        else:
            flawed_total += 1
            is_pass = (act_sup == False)
            if is_pass:
                flawed_caught += 1

        verdict_str = "PASS" if is_pass else "FAIL"
        lines.append(f"[{iid}] Expected: {'SUPPORTED' if exp_sup else 'FLAWED'} | Judge: {'SUPPORTED' if act_sup else 'UNSUPPORTED'} | Result: {verdict_str} ({latency}s)")
        lines.append(f"  Category: {cat}")
        lines.append(f"  Question: {q}")
        lines.append(f"  Answer  : {ans}")
        lines.append(f"  Context : [{chunk_name}]")
        if not exp_sup:
            lines.append(f"  Flaw    : {flaw_desc}")
        lines.append(f"  Judge Explanation : {explanation}")
        lines.append(f"  Unsupported Claims: {claims}")
        lines.append("-" * 90)

        results.append({
            "id": iid,
            "expected_supported": exp_sup,
            "actual_supported": act_sup,
            "passed": is_pass,
            "category": cat,
            "explanation": explanation,
            "unsupported_claims": claims,
            "latency": latency
        })

    corr_pct = round((correct_marked_supported / correct_total) * 100, 1) if correct_total else 0.0
    flaw_pct = round((flawed_caught / flawed_total) * 100, 1) if flawed_total else 0.0

    lines.append("SUMMARY OF HOLDOUT RUN:")
    lines.append(f"  • Correct-marked-supported: {correct_marked_supported}/{correct_total} ({corr_pct}%) [Target >= 90%]")
    lines.append(f"  • Flawed-caught           : {flawed_caught}/{flawed_total} ({flaw_pct}%) [Target >= 90%]")
    
    passed_gate = (corr_pct >= 90.0 and flaw_pct >= 90.0)
    lines.append(f"  • GATE CHECK              : {'PASSED' if passed_gate else 'FAILED'}")
    lines.append("=" * 90)

    report_text = "\n".join(lines)
    print("\n" + report_text)

    out_file = ROOT / "eval" / "diagnostics" / "judge_holdout_run.txt"
    out_file.write_text(report_text, encoding="utf-8")
    print(f"\n[Saved raw results to {out_file}]")

    if not passed_gate:
        print("\n[!] GATE FAILED (< 90%). STOPPING AS MANDATED.")
        sys.exit(1)
    else:
        print("\n[✓] GATE PASSED. Proceeding to downstream evaluation.")

if __name__ == "__main__":
    main()
