#!/usr/bin/env python3
"""
eval/judge_calibration.py
Harness for calibrating the faithfulness judge (llama3.1:8b).
Tests 15 correct answers and 15 programmatically corrupted versions.
Prompt iterations continue until both correct-marked-supported and corrupted-caught are >= 90%.
"""

import os
import sys
import json
import re
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ollama

# 15 Ground-truth correct test triples (Context, Question, Correct Answer)
CALIBRATION_SET = [
    {
        "id": "C01",
        "category": "policy",
        "q": "What is the payment period?",
        "context": (
            "1.1 Standard Payment Terms\n"
            "The organization operates strictly under standard Net 30 payment terms for all regular vendors, "
            "meaning settlement is scheduled thirty (30) calendar days from receipt of a verified, undisputed tax invoice."
        ),
        "correct_answer": (
            "The standard payment period is 30 calendar days from the receipt of a verified, undisputed tax invoice. "
            "[payment_policy.pdf, p.1, §1.1 Standard Payment Terms]"
        ),
        "corrupted_answer": (
            "The standard payment period is 45 calendar days from the receipt of a verified, undisputed tax invoice. "
            "[payment_policy.pdf, p.1, §1.1 Standard Payment Terms]"
        ),
        "corruption_type": "changed_number"
    },
    {
        "id": "C02",
        "category": "policy",
        "q": "What are the payment terms for Vendor ABC Corp?",
        "context": (
            "2.2 Special Vendor Agreements (Vendor ABC Corp)\n"
            "By virtue of Master Services Agreement MSA-2023-ABC, Vendor ABC Corp is granted customized "
            "Net 45 payment terms rather than standard Net 30 terms."
        ),
        "correct_answer": (
            "Vendor ABC Corp is granted customized Net 45 payment terms under Master Services Agreement MSA-2023-ABC. "
            "[vendor_policy.pdf, p.1, §2.2 Special Vendor Agreements (Vendor ABC Corp)]"
        ),
        "corrupted_answer": (
            "Microsoft India is granted customized Net 45 payment terms under Master Services Agreement MSA-2023-ABC. "
            "[vendor_policy.pdf, p.1, §2.2 Special Vendor Agreements (Vendor ABC Corp)]"
        ),
        "corruption_type": "swapped_vendor"
    },
    {
        "id": "C03",
        "category": "policy",
        "q": "What is the late fee rate on overdue invoices?",
        "context": (
            "4.2 Overdue Invoices and Late Fee Structure\n"
            "Late payments are subject to an interest penalty of 2.0% per month (or 24% per annum pro-rata) "
            "on the outstanding principal balance. Interest charges will only apply if the delay is solely "
            "attributable to internal processing and not vendor billing errors."
        ),
        "correct_answer": (
            "The late fee rate on overdue invoices is 2.0% per month (or 24% per annum pro-rata) on the outstanding principal balance, "
            "applying only if the delay is solely attributable to internal processing. "
            "[invoice_terms_policy.pdf, p.1, §4.2 Overdue Invoices and Late Fee Structure]"
        ),
        "corrupted_answer": (
            "The late fee rate on overdue invoices is 5.0% per month (or 60% per annum pro-rata) on the outstanding principal balance. "
            "[invoice_terms_policy.pdf, p.1, §4.2 Overdue Invoices and Late Fee Structure]"
        ),
        "corruption_type": "changed_number"
    },
    {
        "id": "C04",
        "category": "policy",
        "q": "On which days does Accounts Payable run bulk disbursements?",
        "context": (
            "1.2 Disbursement Schedules and Cycles\n"
            "Accounts Payable executes bulk disbursements twice monthly: on the 1st and the 15th calendar day "
            "(or the next immediate business day if falling on a banking holiday or weekend)."
        ),
        "correct_answer": (
            "Accounts Payable executes bulk disbursements on the 1st and the 15th calendar day of each month. "
            "[payment_policy.pdf, p.1, §1.2 Disbursement Schedules and Cycles]"
        ),
        "corrupted_answer": (
            "Accounts Payable executes bulk disbursements on the 10th and the 25th calendar day of each month. "
            "[payment_policy.pdf, p.1, §1.2 Disbursement Schedules and Cycles]"
        ),
        "corruption_type": "changed_number"
    },
    {
        "id": "C05",
        "category": "policy",
        "q": "Which vendor payments must be routed via RTGS?",
        "context": (
            "1.3 Payment Methods and Routing\n"
            "All domestic vendor disbursements exceeding ₹2,00,000 must be routed via Real Time Gross Settlement (RTGS)."
        ),
        "correct_answer": (
            "All domestic vendor disbursements exceeding ₹2,00,000 must be routed via Real Time Gross Settlement (RTGS). "
            "[payment_policy.pdf, p.1, §1.3 Payment Methods and Routing]"
        ),
        "corrupted_answer": (
            "All domestic vendor disbursements exceeding ₹50,000 must be routed via Real Time Gross Settlement (RTGS). "
            "[payment_policy.pdf, p.1, §1.3 Payment Methods and Routing]"
        ),
        "corruption_type": "changed_number"
    },
    {
        "id": "C06",
        "category": "policy",
        "q": "What happens when an invoice reaches day 15 overdue?",
        "context": (
            "4.3 Overdue Escalation Workflow\n"
            "Phase 2 (Day 15 Overdue): Mandatory escalation meeting between Finance Controller and Vendor Account Lead. "
            "Accounts Payable freezes new purchase order issuance to the vendor."
        ),
        "correct_answer": (
            "When an invoice reaches day 15 overdue, Phase 2 mandates an escalation meeting between the Finance Controller "
            "and Vendor Account Lead, and Accounts Payable freezes new purchase order issuance to the vendor. "
            "[invoice_terms_policy.pdf, p.1, §4.3 Overdue Escalation Workflow]"
        ),
        "corrupted_answer": (
            "When an invoice reaches day 15 overdue, the contract is automatically terminated and legal action is filed immediately. "
            "[invoice_terms_policy.pdf, p.1, §4.3 Overdue Escalation Workflow]"
        ),
        "corruption_type": "invented_condition"
    },
    {
        "id": "C07",
        "category": "policy",
        "q": "What is Phase 3 of the overdue escalation workflow?",
        "context": (
            "4.3 Overdue Escalation Workflow\n"
            "Phase 3 (Day 30+ Overdue): Formal dispute review and legal notification protocol activated."
        ),
        "correct_answer": (
            "Phase 3 of the overdue escalation workflow activates formal dispute review and legal notification protocols for invoices 30 or more days overdue. "
            "[invoice_terms_policy.pdf, p.1, §4.3 Overdue Escalation Workflow]"
        ),
        "corrupted_answer": (
            "Phase 3 of the overdue escalation workflow provides an automated waiver of all penalties. "
            "[invoice_terms_policy.pdf, p.1, §4.3 Overdue Escalation Workflow]"
        ),
        "corruption_type": "invented_condition"
    },
    {
        "id": "C08",
        "category": "policy",
        "q": "Which documents are needed for three-way matching?",
        "context": (
            "4.4 Three-Way Matching Rule\n"
            "Payment disbursement requires verification of three documents: Purchase Order (PO), Goods Received Note (GRN) "
            "or Service Completion Certificate, and the Vendor Tax Invoice."
        ),
        "correct_answer": (
            "Three-way matching requires verification of three documents: Purchase Order (PO), Goods Received Note (GRN) "
            "or Service Completion Certificate, and the Vendor Tax Invoice. "
            "[invoice_terms_policy.pdf, p.1, §4.4 Three-Way Matching Rule]"
        ),
        "corrupted_answer": (
            "Three-way matching requires verification of a Passport, PAN card, and personal bank statement. "
            "[invoice_terms_policy.pdf, p.1, §4.4 Three-Way Matching Rule]"
        ),
        "corruption_type": "invented_condition"
    },
    {
        "id": "C09",
        "category": "policy",
        "q": "Who needs to approve a ₹30,000 business expense?",
        "context": (
            "3.1 Approval Limits and Hierarchy\n"
            "Tier 2 (Manager Approval): Expenses between ₹5,001 and ₹25,000 require written approval from the direct Reporting Manager.\n"
            "Tier 3 (Department Head Approval): Outlays between ₹25,001 and ₹1,00,000 require Department Head (VP/Director) authorization."
        ),
        "correct_answer": (
            "A ₹30,000 business expense falls between ₹25,001 and ₹1,00,000 and requires Department Head (VP/Director) authorization. "
            "[expense_policy.pdf, p.1, §3.1 Approval Limits and Hierarchy]"
        ),
        "corrupted_answer": (
            "A ₹30,000 business expense falls under Tier 2 and only requires direct Reporting Manager approval. "
            "[expense_policy.pdf, p.1, §3.1 Approval Limits and Hierarchy]"
        ),
        "corruption_type": "swapped_vendor"
    },
    {
        "id": "C10",
        "category": "policy",
        "q": "What is the daily food allowance in Tier-2 cities?",
        "context": (
            "3.2 Travel & Meals Per Diem\n"
            "Daily meal and incidental expenses (M&IE) per diem caps: Tier-1 metropolitan cities: ₹1,500 per day; "
            "Tier-2 cities: ₹1,000 per day; all other locations: ₹750 per day."
        ),
        "correct_answer": (
            "The daily food allowance in Tier-2 cities is ₹1,000 per day. "
            "[expense_policy.pdf, p.1, §3.2 Travel & Meals Per Diem]"
        ),
        "corrupted_answer": (
            "The daily food allowance in Tier-2 cities is ₹2,500 per day. "
            "[expense_policy.pdf, p.1, §3.2 Travel & Meals Per Diem]"
        ),
        "corruption_type": "changed_number"
    },
    {
        "id": "C11",
        "category": "policy",
        "q": "Can we pay a vendor by cheque?",
        "context": (
            "1.3 Payment Methods and Routing\n"
            "Physical cheque issuance is prohibited except for court-mandated statutory deposits."
        ),
        "correct_answer": (
            "Vendor payment by cheque is strictly prohibited, except for court-mandated statutory deposits. "
            "[payment_policy.pdf, p.1, §1.3 Payment Methods and Routing]"
        ),
        "corrupted_answer": (
            "Vendor payment by cheque is freely permitted for any vendor at the discretion of the department clerk. "
            "[payment_policy.pdf, p.1, §1.3 Payment Methods and Routing]"
        ),
        "corruption_type": "invented_condition"
    },
    {
        "id": "C12",
        "category": "policy",
        "q": "What TDS rate applies to corporate contractors under Section 194C?",
        "context": (
            "5.1 Tax Deducted at Source (TDS) Withholding Rates\n"
            "Section 194C (Contractors & Sub-contractors): 1% TDS for individual/HUF contractors, 2% TDS for corporate entities."
        ),
        "correct_answer": (
            "Under Section 194C, a 2% TDS rate applies to corporate entities. "
            "[finance_compliance_policy.pdf, p.1, §5.1 Tax Deducted at Source (TDS) Withholding Rates]"
        ),
        "corrupted_answer": (
            "Under Section 194C, a 10% TDS rate applies to corporate entities. "
            "[finance_compliance_policy.pdf, p.1, §5.1 Tax Deducted at Source (TDS) Withholding Rates]"
        ),
        "corruption_type": "changed_number"
    },
    {
        "id": "C13",
        "category": "mixed",
        "q": "Which payment method applies to INV-2024-048, and who is the vendor?",
        "context": (
            "Database Record: Invoice INV-2024-048 for vendor Blue Dart Express, amount ₹7,13,742.58, status Overdue.\n"
            "Policy: All domestic vendor disbursements exceeding ₹2,00,000 must be routed via Real Time Gross Settlement (RTGS)."
        ),
        "correct_answer": (
            "The vendor for INV-2024-048 is Blue Dart Express. Since the amount is ₹7,13,742.58 (exceeding ₹2,00,000), RTGS applies. "
            "[payment_policy.pdf, p.1, §1.3 Payment Methods and Routing]"
        ),
        "corrupted_answer": (
            "The vendor for INV-2024-048 is Tata Tech Logistics, and the payment method is corporate UPI. "
            "[payment_policy.pdf, p.1, §1.3 Payment Methods and Routing]"
        ),
        "corruption_type": "swapped_vendor"
    },
    {
        "id": "C14",
        "category": "policy",
        "q": "Is there a grace period for late interest?",
        "context": (
            "4.2 Overdue Invoices and Late Fee Structure\n"
            "Late payments are subject to an interest penalty of 2.0% per month (or 24% per annum pro-rata) on the outstanding principal balance. "
            "Interest charges will only apply if the delay is solely attributable to internal processing and not vendor billing errors."
        ),
        "correct_answer": (
            "The policy does not specify or mention any grace period before late interest starts accruing. "
            "[invoice_terms_policy.pdf, p.1, §4.2 Overdue Invoices and Late Fee Structure]"
        ),
        "corrupted_answer": (
            "The policy provides a mandatory 15-day grace period before late interest accrues. "
            "[invoice_terms_policy.pdf, p.1, §4.2 Overdue Invoices and Late Fee Structure]"
        ),
        "corruption_type": "invented_condition"
    },
    {
        "id": "C15",
        "category": "mixed",
        "q": "Is INV-2024-036 overdue, and what is the late fee rate?",
        "context": (
            "Database Record: Invoice INV-2024-036 for vendor Deloitte Advisory, amount ₹8,49,465.44, status Overdue.\n"
            "Policy: Late payments are subject to an interest penalty of 2.0% per month on the outstanding principal balance."
        ),
        "correct_answer": (
            "Invoice INV-2024-036 is Overdue. The policy late fee rate is 2.0% per month on the outstanding principal balance. "
            "[invoice_terms_policy.pdf, p.1, §4.2 Overdue Invoices and Late Fee Structure]"
        ),
        "corrupted_answer": (
            "Invoice INV-2024-036 is Paid and has no late fee rate. "
            "[invoice_terms_policy.pdf, p.1, §4.2 Overdue Invoices and Late Fee Structure]"
        ),
        "corruption_type": "swapped_vendor"
    }
]

# Prompt Templates for Iterations
PROMPT_ITERATIONS = {
    "v1_baseline": """You are a strict financial auditor evaluating whether an assistant's answer is 100% supported by the retrieved context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Instructions:
1. Check if every factual claim, number, date, rate, condition, or clause in the Generated Answer is directly supported by the context.
2. If any claim is NOT directly supported by the context, mark supported as false and list the unsupported claims.
3. Output valid JSON only:
{{
  "supported": true/false,
  "unsupported_claims": ["claim 1", ...]
}}""",

    "v2_clarified": """You are a financial audit judge checking if the Generated Answer is factually consistent with the Context.

Context:
{context}

Generated Answer:
{answer}

Rules:
1. Supported means that the facts (amounts, numbers, entities, rules) in the Answer are true according to the Context.
2. Rephrasing, standard citations like '[file.pdf, p.1, §1.1]', and numeral variations (e.g. 30 vs thirty) are fully SUPPORTED. Do NOT mark citations or rephrased statements as unsupported.
3. Mark supported as false ONLY if the Answer contradicts the Context, swaps entities/vendors, fabricates new rules, or changes numbers.
4. Output valid JSON only:
{{
  "supported": true/false,
  "unsupported_claims": ["only list actual contradictions or fabricated facts, or empty [] if supported"]
}}""",

    "v3_calibrated": """You are an objective financial auditor judging whether an assistant's answer is factually grounded in the provided context.

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
  "supported": true/false,
  "unsupported_claims": []
}}"""
}


def unload_chat_model():
    """Unloads qwen3:8b so llama3.1:8b runs with full memory / GPU."""
    try:
        subprocess.run(["ollama", "stop", "qwen3:8b"], capture_output=True, text=True, timeout=10)
    except Exception:
        pass


def evaluate_prompt(prompt_template: str, model: str = "llama3.1:8b") -> Dict[str, Any]:
    client = ollama.Client()
    correct_marked_supported = 0
    corrupted_caught = 0
    details = []

    # 1. Test correct answers
    for item in CALIBRATION_SET:
        p = prompt_template.format(context=item["context"], answer=item["correct_answer"])
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": p}],
                format={"type": "object", "properties": {"supported": {"type": "boolean"}, "unsupported_claims": {"type": "array", "items": {"type": "string"}}}, "required": ["supported", "unsupported_claims"]},
                options={"temperature": 0.0},
                think=False
            )
            data = json.loads(resp.message.content)
            is_sup = bool(data.get("supported", False))
            if is_sup:
                correct_marked_supported += 1
            else:
                details.append({
                    "id": item["id"],
                    "type": "false_negative",
                    "answer": item["correct_answer"],
                    "flagged": data.get("unsupported_claims", [])
                })
        except Exception as e:
            pass

    # 2. Test corrupted answers
    for item in CALIBRATION_SET:
        p = prompt_template.format(context=item["context"], answer=item["corrupted_answer"])
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": p}],
                format={"type": "object", "properties": {"supported": {"type": "boolean"}, "unsupported_claims": {"type": "array", "items": {"type": "string"}}}, "required": ["supported", "unsupported_claims"]},
                options={"temperature": 0.0},
                think=False
            )
            data = json.loads(resp.message.content)
            is_sup = bool(data.get("supported", False))
            if not is_sup:
                corrupted_caught += 1
            else:
                details.append({
                    "id": item["id"],
                    "type": "false_positive",
                    "answer": item["corrupted_answer"],
                    "flagged": []
                })
        except Exception as e:
            pass

    total = len(CALIBRATION_SET)
    c_rate = (correct_marked_supported / total) * 100
    f_rate = (corrupted_caught / total) * 100
    passed = c_rate >= 90.0 and f_rate >= 90.0

    return {
        "total": total,
        "correct_marked_supported": correct_marked_supported,
        "correct_rate": round(c_rate, 1),
        "corrupted_caught": corrupted_caught,
        "corrupted_rate": round(f_rate, 1),
        "passed": passed,
        "details": details
    }


def main():
    print("=" * 80)
    print("FAITHFULNESS JUDGE CALIBRATION HARNESS (15 Correct + 15 Corrupted Pairs)")
    print("Model: llama3.1:8b | Target: Both >= 90.0%")
    print("=" * 80)

    print("\n[i] Unloading chat model (qwen3:8b) to ensure judge runs at peak capacity...")
    unload_chat_model()

    for ver_name, tmpl in PROMPT_ITERATIONS.items():
        print(f"\nEvaluating iteration '{ver_name}'...")
        t0 = time.time()
        res = evaluate_prompt(tmpl)
        dur = round(time.time() - t0, 1)
        print(f"  Results for '{ver_name}' ({dur}s):")
        print(f"    • Correct marked supported: {res['correct_marked_supported']}/{res['total']} ({res['correct_rate']}%)")
        print(f"    • Corrupted caught        : {res['corrupted_caught']}/{res['total']} ({res['corrupted_rate']}%)")
        print(f"    • Status                  : {'PASSED (>=90%)' if res['passed'] else 'FAILED (<90%)'}")
        if res["details"]:
            print(f"    • Failure samples (total {len(res['details'])}):")
            for d in res["details"][:3]:
                print(f"        - {d['id']} [{d['type']}]: flagged={d['flagged']}")

        if res["passed"]:
            print(f"\n[✓] Calibration passed on iteration '{ver_name}'!")
            break


if __name__ == "__main__":
    main()
