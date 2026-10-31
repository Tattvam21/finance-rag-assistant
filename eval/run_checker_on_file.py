#!/usr/bin/env python3
"""
eval/run_checker_on_file.py
Runs the deterministic checker on judge_holdout format files (id, question, answer, expected_supported).
Context is extracted directly from the policy PDFs in data/policies using the exact section
splitting logic from eval/make_manual_audit.py (not from the database).

Usage:
  python eval/run_checker_on_file.py --file eval/judge_holdout.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.deterministic_guard import (
    check_deterministic_faithfulness,
    extract_numeric_concepts,
    is_percentage_in_context,
    is_count_in_context,
    are_amounts_equivalent
)
from eval.make_manual_audit import pdf_text, split_sections, CITE_RE

def load_pdf_sections(pdf_dir: Path):
    secs = {}
    for p in sorted(pdf_dir.glob("*.pdf")):
        secs[p.name] = split_sections(pdf_text(p))
    return secs

def get_matched_tokens(ans: str, chunks: list, q: str):
    full_ctx = "\n".join(c.get("content", "") for c in chunks) + "\n" + q
    ans_concepts = extract_numeric_concepts(ans)
    ctx_concepts = extract_numeric_concepts(full_ctx)
    matched = []
    for k, rep in ans_concepts.items():
        if k in ctx_concepts:
            matched.append(rep)
        elif k.startswith("pct_") and is_percentage_in_context(float(k.split("_")[1]), full_ctx):
            matched.append(rep)
        elif (k.startswith("count_") or k.startswith("spelled_")) and is_count_in_context(int(k.split("_")[1]), full_ctx):
            matched.append(rep)
        elif k.startswith("val_"):
            val_amt = float(k.split("_")[1])
            for ck in ctx_concepts.keys():
                if ck.startswith("val_") and are_amounts_equivalent(val_amt, float(ck.split("_")[1])):
                    matched.append(rep)
                    break
    return matched

def run_checker(file_path: Path, pdf_dir: Path):
    if not file_path.exists():
        sys.exit(f"File not found: {file_path}")

    secs = load_pdf_sections(pdf_dir)
    data = json.loads(file_path.read_text(encoding="utf-8"))

    flawed_total = 0
    flawed_caught = 0
    correct_total = 0
    correct_false_flags = 0

    output_lines = []
    header = f"=== RUNNING DETERMINISTIC CHECKER ON {file_path.name} ({len(data)} items) ==="
    output_lines.append(header)
    output_lines.append("-" * len(header))

    for item in data:
        qid = item.get("id")
        exp = item.get("expected_supported")
        ans = item.get("answer", "")
        q = item.get("question", "")

        cites = CITE_RE.findall(ans)
        chunks = []
        for f, n in dict.fromkeys(cites):
            content = secs.get(f, {}).get(n, "")
            chunks.append({
                "source_file": f,
                "section": n,
                "content": content
            })

        res = check_deterministic_faithfulness(ans, chunks, question=q)
        matched_tokens = get_matched_tokens(ans, chunks, q)
        is_flagged = not res.get("passed", True)
        flags = res.get("flags", [])

        if exp is False:
            flawed_total += 1
            if is_flagged:
                flawed_caught += 1
        else:
            correct_total += 1
            if is_flagged:
                correct_false_flags += 1

        status_str = "FLAGGED" if is_flagged else "PASSED"
        output_lines.append(f"[{qid}] Expected Supported: {exp} | Status: {status_str}")
        output_lines.append(f"  Flags: {flags}")
        output_lines.append(f"  Tokens Matched: {matched_tokens}")
        output_lines.append("")

    summary_lines = [
        "=" * 60,
        "DETERMINISTIC CHECKER SUMMARY:",
        f"  • Total items:                         {len(data)}",
        f"  • Flawed items caught:                 {flawed_caught}/{flawed_total} ({flawed_caught/flawed_total*100:.1f}%)" if flawed_total else "  • Flawed items: 0",
        f"  • Correct items with false flags:      {correct_false_flags}/{correct_total} ({correct_false_flags/correct_total*100:.1f}%)" if correct_total else "  • Correct items: 0",
        "=" * 60
    ]
    output_lines.extend(summary_lines)

    result_text = "\n".join(output_lines)
    print(result_text)
    return result_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to holdout JSON file")
    ap.add_argument("--pdf-dir", default="data/policies", help="Directory containing policy PDFs")
    ap.add_argument("--out", default=None, help="Optional output path to save report")
    args = ap.parse_args()

    txt = run_checker(Path(args.file), Path(args.pdf_dir))
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(txt + "\n", encoding="utf-8")
        print(f"\nSaved report to {out_p}")

if __name__ == "__main__":
    main()
