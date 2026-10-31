#!/usr/bin/env python3
"""
eval/corruption_test.py
Reads the real answers for 10 evaluation questions from Config E result JSONs:
  Dev: P01, P03, P05, P07, P11, P12, P14
  Held-out: H01, H03, HM01
Applies 3 programmatic corruptions to each:
  1. Change a number
  2. Swap a unit
  3. Change a tier or phase
Runs the deterministic checker and saves full per-answer output to eval/diagnostics/corruption_test.txt.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.utils.deterministic_guard import check_deterministic_faithfulness

DEV_RESULT = ROOT / "eval" / "results" / "20261017_104757_cfg_e_final_dev.json"
HO_RESULT = ROOT / "eval" / "results" / "20261017_105046_cfg_e_final_heldout.json"
OUT_PATH = ROOT / "eval" / "diagnostics" / "corruption_test.txt"

TARGET_IDS = ["P01", "P03", "P05", "P07", "P11", "P12", "P14", "H01", "H03", "HM01"]

def corrupt_number(ans: str, qid: str) -> str:
    """Mutates a number/percentage in the answer."""
    if qid == "P01":
        return ans.replace("thirty (30)", "forty-five (45)")
    elif qid == "P03":
        return ans.replace("2.0%", "3.5%").replace("24%", "42%")
    elif qid == "P05":
        return ans.replace("₹2,00,000", "₹3,50,000")
    elif qid == "P07":
        return ans.replace("day 15", "day 25")
    elif qid == "P11":
        return ans.replace("₹1,00,000", "₹1,75,000")
    elif qid == "P12":
        return ans.replace("₹6,500", "₹8,500")
    elif qid == "P14":
        return ans.replace("10%", "15%")
    elif qid == "H01":
        return ans.replace("₹30,000", "₹45,000")
    elif qid == "H03":
        # H03 has no numbers, add an invented threshold
        return ans.replace("statutory deposits.", "statutory deposits exceeding ₹50,000.")
    elif qid == "HM01":
        # Add an ungrounded fee number
        return ans.replace("Real Time Gross Settlement (RTGS).", "Real Time Gross Settlement (RTGS) with a flat fee of ₹250.")
    return ans

def corrupt_unit(ans: str, qid: str) -> str:
    """Swaps units (e.g. calendar days -> business days, per month -> per annum)."""
    if "calendar days" in ans:
        return ans.replace("calendar days", "business days")
    elif "per month" in ans:
        return ans.replace("per month", "per annum")
    elif "per night" in ans:
        return ans.replace("per night", "per month")
    elif "day 15" in ans:
        return ans.replace("day 15", "month 15")
    elif qid == "P14":
        return ans.replace("10%", "10% per business quarter")
    elif qid == "H01":
        return ans.replace("business expense", "monthly recurring expense")
    elif qid == "H03":
        return ans.replace("statutory deposits.", "statutory deposits payable within 10 business days.")
    elif qid == "HM01":
        return ans.replace("Real Time Gross Settlement (RTGS)", "RTGS within 5 business days")
    return ans + " calculated per business day"

def corrupt_tier_phase(ans: str, qid: str) -> str:
    """Changes tier or phase levels."""
    if qid == "P01":
        return ans.replace("Net 30", "Phase 1 Net 30")
    elif qid == "P03":
        return ans.replace("overdue invoices", "Phase 1 overdue invoices (escalation phase 1)")
    elif qid == "P05":
        return ans.replace("disbursements exceeding ₹2,00,000", "disbursements exceeding ₹2,00,000 (falling under Tier 1)")
    elif qid == "P07":
        return ans.replace("reaches day 15 overdue", "reaches Phase 3 (day 15 overdue)")
    elif qid == "P11":
        return ans.replace("The Chief Financial Officer (CFO)", "Under Tier 2 rules, the Chief Financial Officer (CFO)")
    elif qid == "P12":
        return ans.replace("Tier-1 cities", "Tier-4 cities")
    elif qid == "P14":
        return ans.replace("Section 194J", "Tier 2 Section 194J")
    elif qid == "H01":
        return ans.replace("The Department Head (VP/Director)", "Under Tier 1, the Department Head (VP/Director)")
    elif qid == "H03":
        return ans.replace("Physical cheque issuance", "Under Tier 3, physical cheque issuance")
    elif qid == "HM01":
        return ans.replace("Real Time Gross Settlement (RTGS)", "Tier 1 IMPS")
    return ans

def main():
    dev_rows = json.loads(DEV_RESULT.read_text(encoding="utf-8")).get("rows", [])
    ho_rows = json.loads(HO_RESULT.read_text(encoding="utf-8")).get("rows", [])

    rows_by_id = {}
    for r in dev_rows:
        rows_by_id[r["id"]] = r
    for r in ho_rows:
        rows_by_id[r["id"]] = r

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("PROGRAMMATIC CORRUPTION TEST ON 10 REAL CONFIG E ANSWERS")
    output_lines.append("=" * 80)

    total_tests = 0
    total_flagged = 0

    for qid in TARGET_IDS:
        r = rows_by_id.get(qid)
        if not r:
            sys.exit(f"Target question {qid} not found in result JSONs")

        orig_ans = (r.get("final_answer") or r.get("answer", "")).strip()
        chunks = r.get("retrieved_chunk_texts", [])
        tool_results = r.get("tool_results", [])
        q = r.get("q", "")

        output_lines.append(f"\n[{qid}] Question: {q}")
        output_lines.append("  ORIGINAL ANSWER:")
        for line in orig_ans.split("\n"):
            output_lines.append(f"    {line}")
        output_lines.append("-" * 60)

        corruptions = [
            ("1. Change Number", corrupt_number(orig_ans, qid)),
            ("2. Swap Unit", corrupt_unit(orig_ans, qid)),
            ("3. Change Tier/Phase", corrupt_tier_phase(orig_ans, qid)),
        ]

        for c_name, c_ans in corruptions:
            total_tests += 1
            res = check_deterministic_faithfulness(c_ans, chunks, tool_results, question=q)
            flagged = not res.get("passed", True)
            if flagged:
                total_flagged += 1

            status_str = "FLAGGED" if flagged else "MISSED"
            output_lines.append(f"  • {c_name} [{status_str}]:")
            output_lines.append(f"    Corrupted Text: {c_ans.replace(chr(10), ' ')}")
            output_lines.append(f"    Flags: {res.get('flags', [])}")

    summary_block = [
        "",
        "=" * 80,
        f"CORRUPTION TEST SUMMARY: Flagged {total_flagged}/{total_tests} ({total_flagged/total_tests*100:.1f}%) programmatically corrupted answers.",
        "=" * 80
    ]
    output_lines.extend(summary_block)

    full_output = "\n".join(output_lines)
    print(full_output)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(full_output + "\n", encoding="utf-8")
    print(f"\nSaved full per-answer corruption report to {OUT_PATH}")

if __name__ == "__main__":
    main()
