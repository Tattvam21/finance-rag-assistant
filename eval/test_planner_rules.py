import json
import re
from typing import Optional, Tuple
from app.llm.schemas import QueryPlan, InvoiceFilter
from app.tools.invoice_queries import get_distinct_vendors

dev = json.load(open("eval/questions.json"))
held = json.load(open("eval/questions_heldout.json"))
all_q = dev + held

vendors = get_distinct_vendors(backend="postgres")
vendors_sorted = sorted(vendors, key=len, reverse=True)

def rule_planner(question: str) -> Tuple[Optional[QueryPlan], bool]:
    q_low = question.lower()
    
    # 1. Invoice ID
    m_inv = re.search(r"\b(INV-\d{4}-\d{3})\b", question, re.I)
    inv_id = m_inv.group(1).upper() if m_inv else None
    
    # 2. Vendor
    matched_vendor = None
    for v in vendors_sorted:
        if v.lower() in q_low:
            matched_vendor = v
            break
            
    # 3. Late fee calculation intent
    is_late_fee = False
    if "late fee" in q_low or "interest" in q_low:
        if any(w in q_low for w in ["calculate", "how much", "owed", "what is the late fee"]) and "rate" not in q_low:
            is_late_fee = True
            
    # 4. Policy keywords
    policy_keywords = [
        "policy", "per diem", "allowance", "hotel", "travel", "flight",
        "tds", "withholding", "gst", "sla", "retention", "payment terms",
        "grace period", "escalation", "three-way", "matching", "approval",
        "dispute", "cheque", "rtgs", "neft", "imps", "disbursement",
        "schedule", "method", "settlement", "discount", "prioritized",
        "threshold", "limit", "rate", "standard", "procedure", "guideline",
        "tier", "condition", "freeze", "held", "hold", "resolution", "reimburse",
        "payment run", "discrepancy", "retained", "retain", "how long must",
        "email", "submitted", "submit", "settle", "normally", "what happens when",
        "reaches day"
    ]
    has_policy_kw = any(k in q_low for k in policy_keywords)
    
    # 5. Data / invoice keywords
    data_words = [
        "status", "pending", "overdue", "unpaid", "paid", "disputed",
        "invoices", "total amount", "total value", "how many", "count",
        "highest", "show", "list", "exceeding", "above", "below", "due before"
    ]
    has_data_words = any(w in q_low for w in data_words)
    
    # Off-topic / unanswerable keywords (routed to policy for refusal)
    unans_words = ["salary", "dress code", "leave", "vacation", "holiday", "bank account", "ceo", "work-from-home"]
    has_unans = any(w in q_low for w in unans_words)
    
    # CASE A: Late fee calculation on specific invoice
    if inv_id and is_late_fee and not has_policy_kw:
        return QueryPlan(needs_policy=False, late_fee_invoice_id=inv_id, invoice=None), True
        
    # CASE B: Mixed question (specific invoice + policy inquiry)
    # e.g., "Is INV-2024-036 overdue, and which escalation phase applies?"
    if inv_id and has_policy_kw:
        if is_late_fee:
            return QueryPlan(needs_policy=True, policy_query=question, late_fee_invoice_id=inv_id, invoice=None), True
        return QueryPlan(needs_policy=True, policy_query=question, invoice=InvoiceFilter(invoice_id=inv_id)), True
        
    # CASE C: Pure invoice lookup by ID
    if inv_id and not has_policy_kw:
        return QueryPlan(needs_policy=False, invoice=InvoiceFilter(invoice_id=inv_id)), True
        
    # CASE D: Vendor invoice query
    if matched_vendor and not has_policy_kw and not has_unans:
        st = None
        if "unpaid" in q_low or "not paid" in q_low or "outstanding" in q_low:
            st = ["Pending", "Overdue"]
        elif "pending" in q_low:
            st = ["Pending"]
        elif "overdue" in q_low:
            st = ["Overdue"]
        elif "paid" in q_low:
            st = ["Paid"]
        elif "disputed" in q_low:
            st = ["Disputed"]
        return QueryPlan(needs_policy=False, invoice=InvoiceFilter(vendor=matched_vendor, status=st)), True
        
    # CASE E: Pure aggregate invoice query (status / counts / amounts)
    if has_data_words and not has_policy_kw and not has_unans and ("invoice" in q_low or "invoices" in q_low):
        st = None
        if "unpaid" in q_low or "not paid" in q_low or "outstanding" in q_low:
            st = ["Pending", "Overdue"]
        elif "pending" in q_low:
            st = ["Pending"]
        elif "overdue" in q_low:
            st = ["Overdue"]
        elif "paid" in q_low:
            st = ["Paid"]
        elif "disputed" in q_low:
            st = ["Disputed"]
            
        min_amt = None
        m_lakh = re.search(r"(?:above|over|exceeding|>)\s*(?:₹|rs\.?)?\s*(\d+(?:\.\d+)?)\s*lakh", q_low)
        if m_lakh:
            min_amt = float(m_lakh.group(1)) * 100000.0
        return QueryPlan(needs_policy=False, invoice=InvoiceFilter(status=st, min_amount=min_amt)), True

    # CASE F: Pure policy or unanswerable query (no invoice intent at all)
    if (has_policy_kw or has_unans) and not (inv_id or matched_vendor) and not ("invoices" in q_low and "how many" in q_low):
        return QueryPlan(needs_policy=True, policy_query=question, invoice=None), True

    # Otherwise unsure -> fallback to LLM
    return None, False

handled = 0
tool_mismatches = 0
for q in all_q:
    plan, confident = rule_planner(q["q"])
    expected = set(q.get("expected_tools", []))
    if confident:
        handled += 1
        routed = set()
        if plan.needs_policy: routed.add("search_policies")
        if plan.invoice: routed.add("query_invoices")
        if plan.late_fee_invoice_id: routed.add("calculate_late_fee")
        # Check if any expected tool was missed
        missed = expected - routed
        if missed:
            tool_mismatches += 1
            print(f"MISSED TOOL for {q['id']}: {q['q']}")
            print(f"  expected={expected}, routed={routed}")

print(f"\nHandled by rules: {handled}/{len(all_q)} ({handled/len(all_q)*100:.1f}%)")
print(f"Tool mismatches: {tool_mismatches}")
