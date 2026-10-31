#!/usr/bin/env python3
"""
Compares two eval result JSON files and prints every question whose pass/fail changed.
Usage:
    python eval/compare.py file1.json file2.json
"""
import sys
import json
from pathlib import Path

def compare_results(file1_path: str, file2_path: str):
    p1 = Path(file1_path)
    p2 = Path(file2_path)

    if not p1.exists() or not p2.exists():
        print(f"Error: One or both files do not exist: {file1_path}, {file2_path}")
        sys.exit(1)

    d1 = json.loads(p1.read_text(encoding="utf-8"))
    d2 = json.loads(p2.read_text(encoding="utf-8"))

    rows1 = {r["id"]: r for r in d1.get("rows", [])}
    rows2 = {r["id"]: r for r in d2.get("rows", [])}

    print("=" * 80)
    print(f"COMPARING EVAL RUNS:")
    print(f"  Run 1 (Baseline)    : {p1.name} (Tag: {d1.get('tag')})")
    print(f"  Run 2 (After Fixes) : {p2.name} (Tag: {d2.get('tag')})")
    print("=" * 80)

    all_ids = sorted(set(rows1.keys()) | set(rows2.keys()))
    improved = []
    regressed = []
    same_fail = []

    for qid in all_ids:
        r1 = rows1.get(qid)
        r2 = rows2.get(qid)

        p1_pass = r1.get("passed", False) if r1 else False
        p2_pass = r2.get("passed", False) if r2 else False

        if not p1_pass and p2_pass:
            improved.append((qid, r1, r2))
        elif p1_pass and not p2_pass:
            regressed.append((qid, r1, r2))
        elif not p1_pass and not p2_pass:
            same_fail.append((qid, r1, r2))

    print(f"\nSUMMARY OF STATUS CHANGES:")
    print(f"  • Regressions (Pass -> Fail) : {len(regressed)}")
    print(f"  • Improvements (Fail -> Pass): {len(improved)}")
    print(f"  • Unchanged Fails            : {len(same_fail)}")

    if regressed:
        print("\n" + "=" * 80)
        print("REGRESSIONS (PASSED IN BASELINE, FAILED IN AFTER_FIXES):")
        print("=" * 80)
        for qid, r1, r2 in regressed:
            print(f"\n[{qid}] Category: {r2.get('category')} | Q: {r2.get('q')}")
            print(f"  Fail reasons in Run 2:")
            for f in r2.get("fails", []):
                print(f"    · {f}")
            print(f"  Tools called in Run 1: {r1.get('tools')} -> Run 2: {r2.get('tools')}")
            print(f"  Retrieved chunks in Run 2: {r2.get('retrieved')}")
            print(f"  Model Answer in Run 2:")
            print(f"    \"{r2.get('answer')}\"")

    if improved:
        print("\n" + "=" * 80)
        print("IMPROVEMENTS (FAILED IN BASELINE, PASSED IN AFTER_FIXES):")
        print("=" * 80)
        for qid, r1, r2 in improved:
            print(f"\n[{qid}] Category: {r2.get('category')} | Q: {r2.get('q')}")
            print(f"  Previous Failures in Run 1: {r1.get('fails')}")
            print(f"  Tools called in Run 2: {r2.get('tools')}")
            print(f"  Retrieved chunks in Run 2: {r2.get('retrieved')}")
            print(f"  Model Answer in Run 2:")
            print(f"    \"{r2.get('answer')}\"")

    if same_fail:
        print("\n" + "=" * 80)
        print("UNCHANGED FAILURES (FAILED IN BOTH RUNS):")
        print("=" * 80)
        for qid, r1, r2 in same_fail:
            print(f"\n[{qid}] Category: {r2.get('category')} | Q: {r2.get('q')}")
            print(f"  Fail reasons in Run 1: {r1.get('fails')}")
            print(f"  Fail reasons in Run 2: {r2.get('fails')}")
            print(f"  Tools called in Run 2: {r2.get('tools')}")
            print(f"  Retrieved chunks in Run 2: {r2.get('retrieved')}")
            print(f"  Model Answer in Run 2:")
            print(f"    \"{r2.get('answer')}\"")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval/compare.py <baseline_json> <after_fixes_json>")
        sys.exit(1)
    compare_results(sys.argv[1], sys.argv[2])
