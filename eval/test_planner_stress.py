import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm.generate import deterministic_rule_planner, get_fallback_plan, generate_query_plan
from app.config import get_ollama_client, CHAT_MODEL

with open("eval/planner_stress.json") as f:
    cases = json.load(f)

total = len(cases)
handled_by_rules = 0
confident_but_wrong = 0
correct_rules = 0
llm_agreement = 0

client = get_ollama_client()

results = []

print(f"Running rules planner stress test on {total} questions...")

for item in cases:
    qid = item["id"]
    q = item["question"]
    exp_policy = item.get("expected_needs_policy", False)
    exp_inv = item.get("expected_has_invoice", False)
    exp_fee = item.get("expected_has_late_fee", bool(item.get("expected_late_fee_id")))

    rule_plan, confident = deterministic_rule_planner(q, backend="postgres")
    
    # Run LLM planner for agreement comparison
    llm_plan, source, _ = generate_query_plan(q, client, CHAT_MODEL, backend="postgres", planner="llm")

    is_handled = confident and rule_plan is not None
    if is_handled:
        handled_by_rules += 1

    # Check rule correctness
    rule_correct = True
    if rule_plan:
        if rule_plan.needs_policy != exp_policy:
            rule_correct = False
        if bool(rule_plan.invoice) != exp_inv:
            rule_correct = False
        if bool(rule_plan.late_fee_invoice_id) != exp_fee:
            rule_correct = False
        if "expected_invoice_id" in item and rule_plan.invoice:
            if rule_plan.invoice.invoice_id != item["expected_invoice_id"]:
                rule_correct = False
        if "expected_late_fee_id" in item:
            if rule_plan.late_fee_invoice_id != item["expected_late_fee_id"]:
                rule_correct = False
        if "expected_vendor" in item and rule_plan.invoice:
            if rule_plan.invoice.vendor != item["expected_vendor"]:
                rule_correct = False
        if "expected_status" in item and rule_plan.invoice:
            if rule_plan.invoice.status != item["expected_status"]:
                rule_correct = False

    if is_handled and not rule_correct:
        confident_but_wrong += 1

    if rule_correct:
        correct_rules += 1

    # Check LLM agreement with expected
    llm_agrees_with_expected = (
        llm_plan.needs_policy == exp_policy
        and bool(llm_plan.invoice) == exp_inv
        and bool(llm_plan.late_fee_invoice_id) == exp_fee
    )
    if "expected_status" in item and llm_plan.invoice:
        if llm_plan.invoice.status != item["expected_status"]:
            llm_agrees_with_expected = False

    if llm_agrees_with_expected:
        llm_agreement += 1

    results.append({
        "id": qid,
        "question": q,
        "dev_duplicate": item.get("dev_duplicate", False),
        "confident": confident,
        "rule_plan": rule_plan.model_dump() if rule_plan else None,
        "llm_plan": llm_plan.model_dump(),
        "rule_correct": rule_correct,
        "llm_agrees": llm_agrees_with_expected
    })

# Compute metrics without dev_duplicate
non_dupes = [r for r in results if not r["dev_duplicate"]]
total_nd = len(non_dupes)
handled_nd = sum(1 for r in non_dupes if r["confident"] and r["rule_plan"] is not None)
cbw_nd = sum(1 for r in non_dupes if r["confident"] and not r["rule_correct"])
correct_rules_nd = sum(1 for r in non_dupes if r["rule_correct"])
llm_agree_nd = sum(1 for r in non_dupes if r["llm_agrees"])

share_handled = (handled_by_rules / total) * 100
llm_agreement_share = (llm_agreement / total) * 100
share_handled_nd = (handled_nd / total_nd) * 100
llm_agree_nd_share = (llm_agree_nd / total_nd) * 100

summary_lines = [
    "============================================================",
    "RULES PLANNER STRESS TEST RESULTS",
    "============================================================",
    f"ALL TEST CASES (including dev duplicates):  {total}",
    f"  Handled by Rules:          {handled_by_rules}/{total} ({share_handled:.1f}%)",
    f"  Confident-but-Wrong:       {confident_but_wrong} (TARGET: 0)",
    f"  Rules Correct:             {correct_rules}/{total} ({(correct_rules/total)*100:.1f}%)",
    f"  LLM Agreement:             {llm_agreement}/{total} ({llm_agreement_share:.1f}%)",
    "------------------------------------------------------------",
    f"EXCLUDING DEV DUPLICATES:                   {total_nd}",
    f"  Handled by Rules:          {handled_nd}/{total_nd} ({share_handled_nd:.1f}%)",
    f"  Confident-but-Wrong:       {cbw_nd} (TARGET: 0)",
    f"  Rules Correct:             {correct_rules_nd}/{total_nd} ({(correct_rules_nd/total_nd)*100:.1f}%)",
    f"  LLM Agreement:             {llm_agree_nd}/{total_nd} ({llm_agree_nd_share:.1f}%)",
    "============================================================"
]
summary_text = "\n".join(summary_lines) + "\n"
print(summary_text)

with open("eval/diagnostics/planner_stress_results.txt", "w") as f:
    f.write(summary_text)
    f.write("\nDetailed Breakdown:\n")
    for r in results:
        f.write(f"[{r['id']}] Dupe: {r['dev_duplicate']} | Confident: {r['confident']} | Rule Correct: {r['rule_correct']} | LLM Agree: {r['llm_agrees']}\n")
        f.write(f"  Q: {r['question']}\n")
        f.write(f"  Rule: {json.dumps(r['rule_plan'])}\n")
        f.write(f"  LLM:  {json.dumps(r['llm_plan'])}\n\n")
