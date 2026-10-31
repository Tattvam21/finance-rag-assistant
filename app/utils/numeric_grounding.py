import re
import math
from typing import List, Dict, Any, Optional, Tuple

def parse_inr_amount(text: str) -> Optional[float]:
    """
    Parses a single currency string or number into a float.
    Handles '₹30,000', '1.5 lakh', '2 crore', '713742.58', etc.
    """
    if not text:
        return None
    s = text.lower().strip()
    multiplier = 1.0
    if "crore" in s or " cr" in s:
        multiplier = 10000000.0
        s = re.sub(r"\b(crore|cr)\b", "", s).strip()
    elif "lakh" in s or "lac" in s:
        multiplier = 100000.0
        s = re.sub(r"\b(lakh|lac)\b", "", s).strip()

    s = s.replace("₹", "").replace("rs.", "").replace("rs", "").replace("inr", "").replace(",", "").strip()
    try:
        val = float(s)
        return val * multiplier
    except ValueError:
        return None

def format_clean_amount(val: float) -> str:
    """Formats float without scientific notation or unnecessary trailing zeros."""
    if val == int(val):
        return f"{int(val)}"
    return f"{val:.2f}".rstrip("0").rstrip(".")

def extract_amounts_from_query(query: str) -> List[float]:
    """
    Extracts all distinct monetary amounts from a user query or text.
    """
    amounts = set()
    # Pattern 1: ₹ or Rs or INR followed by numbers (e.g. ₹30,000 or Rs 1,00,000 or INR 500)
    p1 = re.findall(r"(?:₹|rs\.?|inr)\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", query, flags=re.IGNORECASE)
    for m in p1:
        val = parse_inr_amount(m)
        if val is not None and val > 0:
            amounts.add(val)

    # Pattern 2: Numbers followed by lakh or crore (e.g. 1.5 lakh, 2 crore, 30 lakh)
    p2 = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(lakh|lac|crore|cr)\b", query, flags=re.IGNORECASE)
    for num_str, unit_str in p2:
        val = parse_inr_amount(f"{num_str} {unit_str}")
        if val is not None and val > 0:
            amounts.add(val)

    # Pattern 3: Standalone numbers with commas
    p3 = re.findall(r"\b([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?)\b", query)
    for m in p3:
        val = parse_inr_amount(m)
        if val is not None and val > 0 and val != 2024:
            amounts.add(val)

    # Standalone plain numbers >= 1000 that aren't years or IDs
    p4 = re.findall(r"(?<![a-zA-Z0-9\-_])([1-9][0-9]{3,})(?![a-zA-Z0-9\-_])", query)
    for m in p4:
        val = parse_inr_amount(m)
        if val is not None and val > 0 and val not in (2024, 2023, 2025):
            amounts.add(val)

    return sorted(list(amounts))

def extract_numbers_from_tool_results(tool_results: List[Any]) -> Tuple[List[float], List[int]]:
    """
    Extracts invoice amounts (INR) and overdue days from structured tool results.
    """
    amounts = set()
    overdue_days = set()

    if not tool_results:
        return [], []

    for res in tool_results:
        if isinstance(res, dict):
            # Invoice queries output
            if "invoices" in res and isinstance(res["invoices"], list):
                for inv in res["invoices"]:
                    if isinstance(inv, dict) and "amount_inr" in inv:
                        try:
                            amounts.add(float(inv["amount_inr"]))
                        except (ValueError, TypeError):
                            pass
            if "total_amount" in res and res["total_amount"] is not None:
                try:
                    amounts.add(float(res["total_amount"]))
                except (ValueError, TypeError):
                    pass
            # Single invoice or late fee output
            if "amount_inr" in res:
                try:
                    amounts.add(float(res["amount_inr"]))
                except (ValueError, TypeError):
                    pass
            if "principal" in res:
                try:
                    amounts.add(float(res["principal"]))
                except (ValueError, TypeError):
                    pass
            if "days_overdue" in res:
                try:
                    overdue_days.add(int(res["days_overdue"]))
                except (ValueError, TypeError):
                    pass

    return sorted(list(amounts)), sorted(list(overdue_days))

class PolicyRangeRule:
    def __init__(
        self,
        raw_clause: str,
        sentence: str,
        min_val: float,
        max_val: float,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        label: str = "",
        rule_type: str = "amount",
        source_chunk_id: Optional[str] = None
    ):
        if min_val > max_val:
            raise ValueError(f"Lower bound ({min_val}) cannot be greater than upper bound ({max_val})")
        self.raw_clause = raw_clause
        self.sentence = sentence.strip()
        self.min_val = min_val
        self.max_val = max_val
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive
        self.label = label
        self.rule_type = rule_type
        self.source_chunk_id = source_chunk_id or ""

    def contains(self, val: float) -> bool:
        if self.min_inclusive:
            if val < self.min_val:
                return False
        else:
            if val <= self.min_val:
                return False

        if self.max_inclusive:
            if val > self.max_val:
                return False
        else:
            if val >= self.max_val:
                return False

        return True

def parse_policy_ranges_from_chunks(chunks: List[Any]) -> List[PolicyRangeRule]:
    """
    Parses numeric range clauses from policy text chunks:
    - Approval Tiers: Up to 5,000; Between 5,001 and 25,000; Between 25,001 and 1,00,000; Exceeding 1,00,000
    - Payment Methods: Up to 10,000 (UPI); Between 10,000 and 2,00,000 (NEFT/IMPS); Exceeding 2,00,000 (RTGS)
    - Escalation Phases: Day 1 to 14 Overdue; Day 15 Overdue (up to 29); Day 30+ Overdue
    """
    rules: List[PolicyRangeRule] = []

    for c in chunks:
        sec = ""
        src_file = ""
        if isinstance(c, dict):
            content = c.get("content", "")
            src_file = c.get("source_file", "")
            sec = c.get("section", "")
            src_id = f"{src_file} §{sec}".strip() if src_file else ""
        elif isinstance(c, str):
            content = c
            src_id = ""
        else:
            content = ""
            src_id = ""

        # Collapse intra-sentence linebreaks while preserving paragraph separation
        clean_text = re.sub(r"(?<!\n)\n(?!\n)", " ", content)
        # Split on sentence boundaries (. or ; or newline)
        lines = [s.strip() for s in re.split(r"(?<=[.;\n])\s+", clean_text) if s.strip()]

        for line in lines:
            line_low = line.lower()

            # 1. Approval Tiers (§3.1)
            if "tier 1" in line_low or ("up to" in line_low and "5,000" in line):
                m_t1 = re.search(r"(?:up to|upto)\s*(?:₹|rs\.?)?\s*([0-9,]+)", line, re.I)
                if m_t1:
                    high = parse_inr_amount(m_t1.group(1))
                    if high is not None:
                        try:
                            rules.append(PolicyRangeRule(
                                raw_clause=f"up to ₹{m_t1.group(1)}",
                                sentence=line,
                                min_val=0.0,
                                max_val=high,
                                min_inclusive=True,
                                max_inclusive=True,
                                label="Tier 1 Self-approval by employee",
                                rule_type="tier",
                                source_chunk_id=src_id
                            ))
                        except ValueError:
                            pass

            if "tier 2" in line_low or ("between" in line_low and "5,001" in line and "25,000" in line):
                m_t2 = re.search(r"between\s*(?:₹|rs\.?)?\s*([0-9,]+)\s*and\s*(?:₹|rs\.?)?\s*([0-9,]+)", line, re.I)
                if m_t2:
                    low = parse_inr_amount(m_t2.group(1))
                    high = parse_inr_amount(m_t2.group(2))
                    if low is not None and high is not None:
                        try:
                            rules.append(PolicyRangeRule(
                                raw_clause=f"between ₹{m_t2.group(1)} and ₹{m_t2.group(2)}",
                                sentence=line,
                                min_val=low,
                                max_val=high,
                                min_inclusive=True,
                                max_inclusive=True,
                                label="Tier 2 Team Lead or Manager",
                                rule_type="tier",
                                source_chunk_id=src_id
                            ))
                        except ValueError:
                            pass

            if "tier 3" in line_low or ("between" in line_low and "25,001" in line and "1,00,000" in line):
                m_t3 = re.search(r"between\s*(?:₹|rs\.?)?\s*([0-9,]+)\s*and\s*(?:₹|rs\.?)?\s*([0-9,]+)", line, re.I)
                if m_t3:
                    low = parse_inr_amount(m_t3.group(1))
                    high = parse_inr_amount(m_t3.group(2))
                    if low is not None and high is not None:
                        try:
                            rules.append(PolicyRangeRule(
                                raw_clause=f"between ₹{m_t3.group(1)} and ₹{m_t3.group(2)}",
                                sentence=line,
                                min_val=low,
                                max_val=high,
                                min_inclusive=True,
                                max_inclusive=True,
                                label="Tier 3 Department Head",
                                rule_type="tier",
                                source_chunk_id=src_id
                            ))
                        except ValueError:
                            pass

            if "tier 4" in line_low or ("exceeding" in line_low and "1,00,000" in line):
                m_t4 = re.search(r"(?:exceeding|above|more than)\s*(?:₹|rs\.?)?\s*([0-9,]+)", line, re.I)
                if m_t4:
                    low = parse_inr_amount(m_t4.group(1))
                    if low is not None:
                        try:
                            rules.append(PolicyRangeRule(
                                raw_clause=f"exceeding ₹{m_t4.group(1)}",
                                sentence=line,
                                min_val=low,
                                max_val=float("inf"),
                                min_inclusive=False,
                                max_inclusive=False,
                                label="Tier 4 VP or Finance Director",
                                rule_type="tier",
                                source_chunk_id=src_id
                            ))
                        except ValueError:
                            pass

            # 2. Payment Routing Methods (§1.3)
            if "upi" in line_low and re.search(r"up to\s*(?:₹|rs\.?)?\s*10,000", line, re.I):
                try:
                    rules.append(PolicyRangeRule(
                        raw_clause="up to ₹10,000",
                        sentence=line,
                        min_val=0.0,
                        max_val=10000.0,
                        min_inclusive=True,
                        max_inclusive=True,
                        label="UPI rule",
                        rule_type="payment_method",
                        source_chunk_id=src_id
                    ))
                except ValueError:
                    pass

            if ("neft" in line_low or "imps" in line_low) and re.search(r"between\s*(?:₹|rs\.?)?\s*10,000\s*and\s*(?:₹|rs\.?)?\s*2,00,000", line, re.I):
                try:
                    rules.append(PolicyRangeRule(
                        raw_clause="between ₹10,000 and ₹2,00,000",
                        sentence=line,
                        min_val=10000.0,
                        max_val=200000.0,
                        min_inclusive=True,
                        max_inclusive=True,
                        label="NEFT/IMPS rule",
                        rule_type="payment_method",
                        source_chunk_id=src_id
                    ))
                except ValueError:
                    pass

            if "rtgs" in line_low and re.search(r"exceeding\s*(?:₹|rs\.?)?\s*2,00,000", line, re.I):
                try:
                    rules.append(PolicyRangeRule(
                        raw_clause="exceeding ₹2,00,000",
                        sentence=line,
                        min_val=200000.0,
                        max_val=float("inf"),
                        min_inclusive=False,
                        max_inclusive=False,
                        label="RTGS rule",
                        rule_type="payment_method",
                        source_chunk_id=src_id
                    ))
                except ValueError:
                    pass

            # 3. Escalation Phases - Parse any chunk containing "Phase N (Day ...)"
            if "phase" in line_low:
                m_phase = re.search(r"Phase\s*(\d+)\s*\((.*?)\)", line, re.I)
                if m_phase:
                    p_num = m_phase.group(1)
                    clause_inner = m_phase.group(2).strip()
                    if not re.search(r"\bday\b", clause_inner, re.I):
                        continue

                    m_range = re.search(r"(?:Day\s*)?(\d+)\s*(?:to|-)\s*(\d+)", clause_inner, re.I)
                    m_plus = re.search(r"(?:Day\s*)?(\d+)\+", clause_inner, re.I)
                    m_single = re.search(r"(?:Day\s*)?(\d+)", clause_inner, re.I)

                    if m_range:
                        low = float(m_range.group(1))
                        high = float(m_range.group(2))
                        try:
                            rules.append(PolicyRangeRule(
                                raw_clause=clause_inner,
                                sentence=line,
                                min_val=low,
                                max_val=high,
                                min_inclusive=True,
                                max_inclusive=True,
                                label=f"Phase {p_num}",
                                rule_type="overdue_phase",
                                source_chunk_id=src_id
                            ))
                        except ValueError:
                            pass
                    elif m_plus:
                        low = float(m_plus.group(1))
                        try:
                            rules.append(PolicyRangeRule(
                                raw_clause=clause_inner,
                                sentence=line,
                                min_val=low,
                                max_val=float("inf"),
                                min_inclusive=True,
                                max_inclusive=False,
                                label=f"Phase {p_num}",
                                rule_type="overdue_phase",
                                source_chunk_id=src_id
                            ))
                        except ValueError:
                            pass
                    elif m_single:
                        val = float(m_single.group(1))
                        try:
                            rules.append(PolicyRangeRule(
                                raw_clause=clause_inner,
                                sentence=line,
                                min_val=val,
                                max_val=val,
                                min_inclusive=True,
                                max_inclusive=True,
                                label=f"Phase {p_num}",
                                rule_type="overdue_phase",
                                source_chunk_id=src_id
                            ))
                        except ValueError:
                            pass

    return rules

def build_verified_numeric_facts(
    question: str,
    chunks: List[Any],
    tool_results: Optional[List[Any]] = None,
    cited_chunk_ids: Optional[List[int]] = None
) -> Tuple[str, Optional[str]]:
    """
    Builds a VERIFIED NUMERIC FACTS block comparing question/invoice amounts against policy rules.
    Generic mechanism:
    - Facts extracted ONLY from chunks ranked <= 2 or cited by the answer.
    - Every fact is labeled with its source chunk ID.
    - Amounts taken ONLY from question text and from selected invoice rows (max 10 rows, never whole table).
    - Rejects any range where lower bound > upper bound.
    Returns: (facts_block, best_matching_sentence)
    """
    if not chunks:
        return "", None

    # Filter chunks: only rank <= 2 (indices 0 and 1) or explicitly cited chunk IDs
    allowed_indices = {0, 1}
    if cited_chunk_ids:
        for cid in cited_chunk_ids:
            idx = cid - 1
            if 0 <= idx < len(chunks):
                allowed_indices.add(idx)

    scoped_chunks = [c for i, c in enumerate(chunks) if i in allowed_indices]
    rules = parse_policy_ranges_from_chunks(scoped_chunks)
    if not rules:
        return "", None

    # Guard against whole-table scans (e.g. query_invoices({}) returning more than 10 invoices)
    has_bulk_invoices = any(isinstance(t, dict) and (t.get("count", 0) > 10 or len(t.get("invoices", [])) > 10) for t in (tool_results or []))
    if has_bulk_invoices:
        tool_amounts = []
        tool_days = []
    else:
        tool_amounts, tool_days = extract_numbers_from_tool_results(tool_results or [])

    q_amounts = extract_amounts_from_query(question)
    all_amounts = sorted(list(set(q_amounts + tool_amounts)))

    facts_lines: List[str] = []
    best_matching_quote: Optional[str] = None

    # Group rules by (source_chunk_id, rule_type)
    chunk_rules: Dict[Tuple[str, str], List[PolicyRangeRule]] = {}
    for r in rules:
        key = (r.source_chunk_id, r.rule_type)
        chunk_rules.setdefault(key, []).append(r)

    # 1. Compare amounts against amount-based rules (tier and payment_method)
    for amt in all_amounts:
        clean_amt = format_clean_amount(amt)
        for (src_id, r_type), r_list in chunk_rules.items():
            if r_type not in ("tier", "payment_method"):
                continue
            matched = next((r for r in r_list if r.contains(amt)), None)
            if matched:
                non_matched = [r for r in r_list if r != matched]
                not_clauses = [f"'{r.raw_clause}'" for r in non_matched]
                src_prefix = f"[{src_id}] " if src_id else ""
                if matched.rule_type == "payment_method" and matched.min_val >= 200000.0:
                    line_msg = f"{src_prefix}Amount {clean_amt} exceeds 200000 ({matched.label}); NOT in {', '.join(not_clauses)}."
                else:
                    line_msg = f"{src_prefix}Amount {clean_amt} falls in '{matched.raw_clause}' ({matched.label})"
                    if not_clauses:
                        line_msg += f"; NOT in {', '.join(not_clauses)}."
                    else:
                        line_msg += "."
                facts_lines.append(line_msg)
                if not best_matching_quote:
                    best_matching_quote = matched.sentence

    # 2. Compare overdue days against escalation phases
    q_days = [int(m) for m in re.findall(r"\b(\d+)\s*(?:days?\s*overdue|days?\s*past\s*due)\b", question, re.I)]
    q_reach = [int(m) for m in re.findall(r"\breaches\s*day\s*(\d+)\b", question, re.I)]
    q_day_n = [int(m) for m in re.findall(r"\bday\s*(\d+)\b", question, re.I)]
    all_overdue_days = sorted(list(set(tool_days + q_days + q_reach + q_day_n)))

    if all_overdue_days:
        for d in all_overdue_days:
            for (src_id, r_type), r_list in chunk_rules.items():
                if r_type != "overdue_phase":
                    continue
                src_prefix = f"[{src_id}] " if src_id else ""
                p1 = next((r for r in r_list if r.label == "Phase 1"), None)
                p2 = next((r for r in r_list if r.label == "Phase 2"), None)
                p3 = next((r for r in r_list if r.label == "Phase 3"), None)
                # If phases cannot be parsed from chunks, emit no phase facts (no fallback defaults)
                if not (p2 and p3):
                    continue
                p2_day = p2.max_val
                p3_day = p3.min_val

                if d <= 0:
                    facts_lines.append(f"{src_prefix}0 days overdue: Not overdue; no overdue escalation phase applies.")
                elif p2_day < d < p3_day:
                    gap_start = int(p2_day) + 1
                    gap_end = int(p3_day) - 1
                    gap_str = f"{gap_start}-{gap_end}" if gap_start < gap_end else f"{gap_start}"
                    facts_lines.append(f"{src_prefix}Phase 2 was triggered on Day {int(p2_day)}; Phase 3 starts at Day {int(p3_day)}+; the policy does not define days {gap_str} explicitly.")
                    if not best_matching_quote and p2:
                        best_matching_quote = p2.sentence
                else:
                    matched_ph = next((r for r in r_list if r.contains(float(d))), None)
                    if matched_ph:
                        line_msg = f"{src_prefix}{d} days overdue falls in '{matched_ph.raw_clause}' ({matched_ph.label})"
                        facts_lines.append(line_msg)
                        if not best_matching_quote:
                            best_matching_quote = matched_ph.sentence

    if not facts_lines:
        return "", None

    block = (
        "VERIFIED NUMERIC FACTS (DETERMINISTIC SYSTEM GROUNDING):\n"
        + "\n".join(f"- {l}" for l in facts_lines) + "\n"
        "CRITICAL INSTRUCTION: Never attempt to compare numbers or guess range membership. "
        "Strictly apply the verified numeric facts above."
    )
    return block, best_matching_quote
