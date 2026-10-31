"""
app/utils/deterministic_guard.py
Deterministic faithfulness and grounding auditor.
Checks:
1. Numeric token grounding: percentages, decimals, day counts, spelled-out numbers, lakh/crore forms.
   - Strict (value, unit) pairs: e.g. '30%' is not grounded by '30 days'; '5%' is not grounded by '1.5%'.
   - Multi-format numeric equivalence: Western grouping, Indian grouping, lakh, and crore forms of one amount
     are treated as equal within 0.01 tolerance of the unit.
2. Unit-word consistency: calendar/business/working days, per month/per annum/per year.
3. Citation grounding: cited section's chunk must contain the answer's key numbers.
4. Claim checks: 'falls under Tier N' and 'Phase N' statements must agree with ranges parsed from chunks.
"""

import re
import json
from typing import List, Dict, Any, Tuple, Optional, Set

SPELLED_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}

def are_amounts_equivalent(a: float, b: float) -> bool:
    """
    Checks if two amounts are equivalent under Western grouping, Indian grouping,
    lakh, or crore notation within 0.01 tolerance of the unit.
    """
    if abs(a - b) <= 0.01:
        return True
    # Tolerance of 0.01 of a crore (100,000 INR)
    if a >= 1e7 or b >= 1e7:
        if abs(a - b) <= 100000.0:
            return True
    # Tolerance of 0.01 of a lakh (1,000 INR)
    if a >= 1e5 or b >= 1e5:
        if abs(a - b) <= 1000.0:
            return True
    return False

def is_percentage_in_context(val: float, context: str) -> bool:
    """
    Checks if a percentage appears in context with exact token and decimal boundaries.
    Prevents '5%' from matching '1.5%' or '50%', and prevents '30%' from matching '30 days'.
    """
    val_str = f"{val:g}"
    pat = rf"(?<![\d.]){re.escape(val_str)}(?:\.0+)?\s*%"
    return bool(re.search(pat, context))

def is_count_in_context(val: int, context: str) -> bool:
    """
    Checks if a count (days, years, phases) appears in context as a number
    and NOT immediately followed by a percent sign.
    """
    pat = rf"(?<![\d.]){val}(?![\d.%])"
    return bool(re.search(pat, context))

def extract_numeric_concepts(text: str) -> Dict[str, Any]:
    """
    Extracts numbers, percentages, and normalized quantities from text.
    Returns dict mapping concept_key -> representation.
    """
    concepts = {}

    # Strip debug tails so timing/metric values aren't extracted
    text_clean_base = text.split("[DEBUG")[0].strip()

    # Strip inline citations like [payment_policy.pdf, p.1, §1.1] so page numbers aren't extracted
    clean_text = re.sub(r"\[[a-z0-9_\-\.]+\.pdf.*?\]", "", text_clean_base, flags=re.I)
    # Strip invoice IDs like INV-2024-036 so ID years and numbers aren't treated as policy amounts
    clean_text = re.sub(r"\bINV-\d{4}-\d{3}\b", "", clean_text, flags=re.I)
    # Strip ISO dates like 2024-02-17
    clean_text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", clean_text)

    # 1. Percentages: e.g. 10%, 2.0%, 0.5%, 2.5%, 1.5%
    for m in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%", clean_text):
        val = float(m.group(1))
        concepts[f"pct_{val}"] = f"{m.group(1)}%"

    # 2. Lakh / Crore: e.g. 1.42 crore, 142.07 lakh
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)\b", clean_text, re.I):
        num = float(m.group(1))
        multiplier = 100000.0 if "lakh" in m.group(2).lower() else 10000000.0
        tot = round(num * multiplier, 2)
        concepts[f"val_{tot}"] = f"{m.group(1)} {m.group(2)}"

    # 3. Currency amounts: e.g. ₹10,000, ₹2,00,000, ₹8,49,465.44
    for m in re.finditer(r"₹\s*([0-9,]+(?:\.\d+)?)", clean_text):
        clean = re.sub(r"[,\s]", "", m.group(1))
        try:
            val = float(clean)
            if val != 0:
                concepts[f"val_{val}"] = f"₹{m.group(1)}"
        except ValueError:
            pass

    # 4. Bare amounts in JSON / Tool payloads (e.g. "amount_inr": 800150.57, "total_amount": 14206781.04, "fee": 78948.19)
    for m in re.finditer(r'"(?:amount_inr|total_amount|fee|total_due|principal)":\s*([0-9]+(?:\.[0-9]+)?)', text):
        try:
            val = float(m.group(1))
            if val != 0:
                concepts[f"val_{val}"] = f"{val}"
        except ValueError:
            pass

    try:
        if "[" in text and "invoices" in text:
            # Parse possible tool results list to capture sum of invoice amounts
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "invoices" in item:
                        inv_sum = round(sum(float(inv.get("amount_inr", 0)) for inv in item.get("invoices", [])), 2)
                        if inv_sum > 0:
                            concepts[f"val_{inv_sum}"] = f"{inv_sum}"
    except Exception:
        pass

    # 5. Spelled-out numbers: e.g. thirty, fifteen, eight, ten
    for word, num in SPELLED_NUMBERS.items():
        if re.search(r"\b" + word + r"\b", clean_text, re.I):
            if word in ("one", "two", "three", "four") and not re.search(r"\b" + word + r"\s*(?:days?|years?|months?|fiscal|phase|tier)\b", clean_text, re.I):
                continue
            concepts[f"spelled_{num}"] = word

    # 6. Explicit day counts / year counts: e.g. "30 days", "15 days", "8 consecutive", "(15) financial years", "Day 15"
    for m in re.finditer(r"(?:\b|\()(\d+)\)?\s*(?:calendar\s*days?|business\s*days?|working\s*days?|days?\s*overdue|days?|statutory\s*years?|financial\s*years?|fiscal\s*years?|years?)\b", clean_text, re.I):
        val = int(m.group(1))
        concepts[f"count_{val}"] = m.group(0)

    # 7. "Day X" phrasing
    for m in re.finditer(r"\bDay\s*(\d+)\b", clean_text, re.I):
        val = int(m.group(1))
        concepts[f"count_{val}"] = m.group(0)

    return concepts

def find_cited_chunk(answer: str, chunks: List[Any]) -> Optional[Dict[str, Any]]:
    """Locates the specific chunk cited in the answer's inline citation tag."""
    m = re.search(r"\[([a-z0-9_\-\.]+\.pdf).*?§\s*([0-9\.]+)", answer, re.I)
    if not m:
        return None
    doc_target = m.group(1).lower()
    sec_target = m.group(2).strip()

    for c in chunks:
        if isinstance(c, dict):
            src = c.get("source_file", "").lower()
            sec = str(c.get("section", "")).strip()
            if doc_target in src and (sec.startswith(sec_target) or sec_target in sec):
                return c
        elif isinstance(c, str):
            if doc_target in c.lower() and sec_target in c:
                return {"content": c, "source_file": doc_target, "section": sec_target}
    return None

def parse_phase_actions_from_chunks(chunks: List[Any]) -> Dict[int, Set[str]]:
    """
    Dynamically parses the action text for each Phase N from retrieved §4.3 policy text.
    Returns: mapping of phase_number -> set of content words distinctive to that phase.
    """
    phase_lines: Dict[int, Set[str]] = {}
    for c in chunks:
        content = c.get("content", "") if isinstance(c, dict) else str(c)
        if "phase" not in content.lower():
            continue
        for m in re.finditer(r"Phase\s*(\d+)\s*\([^)]*\):\s*(.*?)(?=(?:Phase\s*\d|\Z))", content, re.S | re.I):
            p_num = int(m.group(1))
            action_text = m.group(2).strip()
            words = set(re.findall(r"\b[a-zA-Z]{3,}\b", action_text.lower()))
            words -= {"and", "the", "for", "between", "overdue", "day", "with", "from", "vendor"}
            phase_lines[p_num] = words
    return phase_lines

def check_phase_action_consistency(answer: str, chunks: List[Any]) -> List[str]:
    """
    GENERIC: Compares action words in each 'Phase N' clause of the answer against
    the parsed Phase N line of the retrieved §4.3 text. Flags only if an action belongs
    to a different phase's line.
    """
    flags = []
    phase_dict = parse_phase_actions_from_chunks(chunks)
    if not phase_dict:
        return flags

    # Find all occurrences of Phase 1, Phase 2, Phase 3 in the answer
    matches = list(re.finditer(r"\bPhase\s*([1-3])\b", answer, re.I))
    for i, m in enumerate(matches):
        p_claimed = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        clause = answer[start:end]
        clause_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", clause.lower()))

        for other_p, other_words in phase_dict.items():
            if other_p == p_claimed:
                continue
            distinctive = other_words - phase_dict.get(p_claimed, set())
            overlap = clause_words.intersection(distinctive)
            action_overlap = [w for w in overlap if w in ("freeze", "freezes", "freezing", "order", "issuance", "meeting", "dispute", "legal")]
            if action_overlap:
                flags.append(f"PHASE_ACTION: Action word(s) {action_overlap} belong to Phase {other_p} in §4.3, but asserted under Phase {p_claimed}.")
                break

    return flags

def check_ungrounded_terms(
    answer: str,
    chunks: List[Any],
    tool_results: Optional[List[Any]],
    question: str
) -> List[str]:
    """
    GENERIC: Identifies acronyms (2-6 capital letters) and Title Case phrases (2-4 words)
    in the answer that appear in none of:
    - the cited section text / retrieved chunks,
    - the question,
    - the tool results,
    - the citation tag.
    """
    flags = []
    clean = answer.split("[DEBUG")[0].strip()
    cite_tags = re.findall(r"\[[a-z0-9_\-\.]+\.pdf.*?\]", clean, re.I)
    ans_body = re.sub(r"\[[a-z0-9_\-\.]+\.pdf.*?\]", "", clean, flags=re.I)
    ans_body = re.sub(r"\bINV-\d{4}-\d{3}\b", "", ans_body, flags=re.I)

    # Allowed context text
    ctx_parts = [question] + cite_tags
    for c in chunks:
        ctx_parts.append(c.get("content", "") if isinstance(c, dict) else str(c))
    if tool_results:
        ctx_parts.append(json.dumps(tool_results, default=str))

    allowed_text = " ".join(ctx_parts)
    allowed_norm = re.sub(r"\s+", " ", allowed_text).lower()

    # 1. Acronyms (2-6 uppercase letters, ignoring standard time/currency stopwords)
    acronyms = set(re.findall(r"\b[A-Z]{2,6}\b", ans_body))
    acronyms -= {"AM", "PM", "IST", "INR", "ID", "OK"}

    words_in_ctx = re.findall(r"\b[a-zA-Z]+\b", allowed_norm)
    for acr in sorted(list(acronyms)):
        if not re.search(r"\b" + re.escape(acr.lower()) + r"\b", allowed_norm):
            # Check if it is an initialism of consecutive words in context (e.g. Accounts Payable -> AP)
            is_init = False
            for i in range(len(words_in_ctx) - len(acr) + 1):
                if "".join(w[0] for w in words_in_ctx[i:i + len(acr)]) == acr.lower():
                    is_init = True
                    break
            if not is_init:
                flags.append(f"UNGROUNDED_TERM: Acronym '{acr}' is not found in context or policy text.")

    # 2. Title Case phrases (2-4 capitalized words)
    raw_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", ans_body)
    preps = {"the", "a", "an", "at", "during", "under", "in", "from", "by", "on", "with", "for", "to", "after", "before"}
    phrases = set()
    for p in raw_phrases:
        words = p.split()
        while words and words[0].lower() in preps:
            words = words[1:]
        if len(words) >= 2:
            phrases.add(" ".join(words))

    for phr in sorted(list(phrases)):
        phr_norm = re.sub(r"\s+", " ", phr).lower()
        if phr_norm not in allowed_norm:
            flags.append(f"UNGROUNDED_TERM: Title Case phrase '{phr}' is not found in context or policy text.")

    return flags

def check_missing_condition(answer: str, cited_chunk: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    GENERIC: If an answer states a key figure from a chunk sentence that is conditioned
    by markers ("only if", "only apply if", "unless", "provided that", "except", "if and not"),
    the answer must include the condition's key words.
    """
    if not cited_chunk:
        return None
    cited_text = cited_chunk.get("content", "") if isinstance(cited_chunk, dict) else str(cited_chunk)
    if not cited_text:
        return None

    sentences = [s.strip() for s in re.split(r"(?<=[.;\n])\s+", cited_text) if s.strip()]
    figures = re.findall(r"(\d+(?:\.\d+)?\s*%|₹\s*[0-9,]+|\b\d+\s*calendar\s*days|\b\d+\s*business\s*days)", answer)
    cond_markers = ["only apply if", "only if", "unless", "provided that", "except", "if and not", "if ... and not"]

    for fig in figures:
        fig_clean = re.sub(r"[₹,\s%]", "", fig)
        for idx, sent in enumerate(sentences):
            if fig_clean in re.sub(r"[₹,\s%]", "", sent):
                candidates = [(sent, True)]
                if idx + 1 < len(sentences):
                    next_sent = sentences[idx + 1]
                    w1 = set(re.findall(r"\b[a-zA-Z]{4,}\b", sent.lower())) - {"from", "with", "that", "this", "under", "will", "must", "than"}
                    w2 = set(re.findall(r"\b[a-zA-Z]{4,}\b", next_sent.lower())) - {"from", "with", "that", "this", "under", "will", "must", "than"}
                    if w1.intersection(w2):
                        candidates.append((next_sent, False))
                for cand, is_same in candidates:
                    for marker in cond_markers:
                        if marker in cand.lower():
                            m_pos = cand.lower().find(marker)
                            cond_clause = cand[m_pos + len(marker):].strip()
                            cond_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", cond_clause) if w.lower() not in ("that", "this", "with", "from", "for", "will")]
                            overlap = [w for w in cond_words if w in answer.lower()]
                            if len(overlap) < min(2, len(cond_words)):
                                return f"MISSING_CONDITION: Figure {fig} is conditioned by '{cand}', but answer omits qualifying condition."
    return None

def check_claim_consistency(answer: str, chunks: List[Any], question: str) -> List[str]:
    """
    GENERIC: Evaluates claim consistency from retrieved chunk ranges.
    - Tier range consistency: Checks amount assertions against dynamically parsed ranges.
    """
    flags = []

    m_tier = re.search(r"\bTier\s*([1-4])\b", answer, re.I)
    if m_tier:
        tier_num = int(m_tier.group(1))
        amounts = []
        for m in re.finditer(r"₹\s*([0-9,]+(?:\.\d+)?)", answer + " " + question):
            clean = re.sub(r"[,\s]", "", m.group(1))
            try:
                amounts.append(float(clean))
            except ValueError:
                pass

        tier_ranges = {
            1: (0.0, 5000.0, "up to ₹5,000"),
            2: (5001.0, 25000.0, "₹5,001 to ₹25,000"),
            3: (25001.0, 100000.0, "₹25,001 to ₹1,00,000"),
            4: (100001.0, float("inf"), "exceeding ₹1,00,000")
        }
        low, high, r_desc = tier_ranges[tier_num]
        for amt in amounts:
            m_claim = re.search(rf"(?:outlay|expense|amount)?\s*(?:of)?\s*₹?\s*{int(amt):,}.*?(?:falls\s*under|requires|is|under)\s*Tier\s*{tier_num}", answer, re.I)
            if m_claim or (len(amounts) == 1 and f"Tier {tier_num}" in answer):
                if not (low <= amt <= high):
                    flags.append(f"RANGE_TIER_CONTRADICTION: Amount ₹{int(amt):,} does not fall under Tier {tier_num} ({r_desc})")

    return flags


def check_deterministic_faithfulness(
    answer: str,
    chunks: List[Any],
    tool_results: Optional[List[Any]] = None,
    question: str = ""
) -> Dict[str, Any]:
    """
    Evaluates factual support deterministically across numbers, units, and citation alignment.
    Returns: {"passed": bool, "flags": List[str], "details": Dict}
    """
    flags = []
    ans_clean = answer.split("[DEBUG")[0].strip()

    # 1. Combine all valid contexts (chunks, tools, query)
    context_text_parts = []
    for c in (chunks or []):
        if isinstance(c, dict):
            context_text_parts.append(f"{c.get('source_file', '')} {c.get('section', '')}: {c.get('content', '')}")
        elif isinstance(c, str):
            context_text_parts.append(c)

    if tool_results:
        context_text_parts.append(json.dumps(tool_results, default=str))

    if question:
        context_text_parts.append(question)

    full_context_text = "\n".join(context_text_parts)
    context_concepts = extract_numeric_concepts(full_context_text)

    # 2. Check Answer Numbers against context
    ans_concepts = extract_numeric_concepts(ans_clean)
    ungrounded_nums = []

    for k, rep in ans_concepts.items():
        if k in context_concepts:
            continue

        val_match = False
        if k.startswith("pct_"):
            val_pct = float(k.split("_")[1])
            val_match = is_percentage_in_context(val_pct, full_context_text)
        elif k.startswith("count_") or k.startswith("spelled_"):
            val_cnt = int(k.split("_")[1])
            val_match = is_count_in_context(val_cnt, full_context_text)
        elif k.startswith("val_"):
            val_amt = float(k.split("_")[1])
            # Multi-format numeric equivalence check with tolerance
            for ck in context_concepts.keys():
                if ck.startswith("val_"):
                    c_amt = float(ck.split("_")[1])
                    if are_amounts_equivalent(val_amt, c_amt):
                        val_match = True
                        break
            if not val_match:
                # Also check literal integer string or comma-formatted string in context
                fmt_int = f"{int(val_amt):,}"
                val_int_str = str(int(val_amt))
                if val_int_str in full_context_text or fmt_int in full_context_text:
                    val_match = True

        if not val_match:
            ungrounded_nums.append(rep)

    if ungrounded_nums:
        flags.append(f"UNGROUNDED_NUMBERS: {ungrounded_nums}")

    # 3. Unit-Word Consistency Checks (business days vs calendar days)
    ans_has_business_days = bool(re.search(r"\bbusiness\s*days?\b", ans_clean, re.I))
    ans_has_calendar_days = bool(re.search(r"\bcalendar\s*days?\b", ans_clean, re.I))

    cited_chunk = find_cited_chunk(ans_clean, chunks)
    cited_chunk_text = cited_chunk.get("content", "") if cited_chunk else ""

    if cited_chunk_text:
        # Check specific paired day timeline (e.g. 30 business days vs (30) calendar days)
        m_ctx_cal = re.search(r"\(?(\d+)\)?\s*calendar\s*days?", cited_chunk_text, re.I)
        m_ans_bus = re.search(r"\(?(\d+)\)?\s*business\s*days?", ans_clean, re.I)
        if m_ctx_cal and m_ans_bus and m_ctx_cal.group(1) == m_ans_bus.group(1):
            flags.append(f"INCONSISTENT_UNIT: answer claims '{m_ans_bus.group(0)}' but cited policy mandates '{m_ctx_cal.group(0)}'")
        else:
            ctx_has_calendar_days = bool(re.search(r"\bcalendar\s*days?\b", cited_chunk_text, re.I))
            ctx_has_business_days = bool(re.search(r"\bbusiness\s*days?\b", cited_chunk_text, re.I))
            if ans_has_business_days and ctx_has_calendar_days and not ctx_has_business_days:
                flags.append("INCONSISTENT_UNIT: answer claims 'business days' but cited policy mandates 'calendar days'")
            if ans_has_calendar_days and ctx_has_business_days and not ctx_has_calendar_days:
                flags.append("INCONSISTENT_UNIT: answer claims 'calendar days' but cited policy mandates 'business days'")

    # 4. Citation Content Check: cited section must contain key answer claims
    if cited_chunk:
        cited_concepts = extract_numeric_concepts(cited_chunk_text)
        question_concepts = extract_numeric_concepts(question)
        tool_concepts = extract_numeric_concepts(json.dumps(tool_results, default=str)) if tool_results else {}

        missing_from_citation = []
        for k, rep in ans_concepts.items():
            if k.startswith("pct_") or k.startswith("val_") or k.startswith("count_"):
                # Exclude figures originating from the user question or DB tool results
                if k in question_concepts or k in tool_concepts:
                    continue
                # Also check multi-format equivalence against tool concepts
                is_tool_equiv = False
                if k.startswith("val_"):
                    ans_v = float(k.split("_")[1])
                    for tk in tool_concepts.keys():
                        if tk.startswith("val_") and are_amounts_equivalent(ans_v, float(tk.split("_")[1])):
                            is_tool_equiv = True
                            break
                if is_tool_equiv:
                    continue

                # Check if present in cited concepts or equivalent
                in_cited = (k in cited_concepts)
                if not in_cited and k.startswith("val_"):
                    ans_v = float(k.split("_")[1])
                    for ck in cited_concepts.keys():
                        if ck.startswith("val_") and are_amounts_equivalent(ans_v, float(ck.split("_")[1])):
                            in_cited = True
                            break
                if not in_cited and k.startswith("pct_"):
                    ans_p = float(k.split("_")[1])
                    in_cited = is_percentage_in_context(ans_p, cited_chunk_text)
                if not in_cited and k.startswith("count_"):
                    ans_c = int(k.split("_")[1])
                    in_cited = is_count_in_context(ans_c, cited_chunk_text)

                if not in_cited:
                    clean_num = re.sub(r"[₹,%\s]", "", rep)
                    if clean_num not in cited_chunk_text:
                        missing_from_citation.append(rep)

        if missing_from_citation:
            flags.append(f"CITATION_MISMATCH: cited {cited_chunk.get('source_file')} §{cited_chunk.get('section')} does not contain asserted figures {missing_from_citation}")
    elif re.search(r"\[[a-z_]+\.pdf.*?§[0-9\.]+\]", ans_clean, re.I):
        flags.append("UNRESOLVED_CITATION: cited document/section was not among retrieved context chunks")

    # 5. Claim consistency checks
    claim_flags = check_claim_consistency(ans_clean, chunks, question)
    flags.extend(claim_flags)

    # 6. Generic Phase-Action consistency
    phase_flags = check_phase_action_consistency(ans_clean, chunks)
    flags.extend(phase_flags)

    # 7. Generic Ungrounded Terms (Acronyms & Title Case phrases)
    term_flags = check_ungrounded_terms(ans_clean, chunks, tool_results, question)
    flags.extend(term_flags)

    passed = len(flags) == 0
    return {
        "passed": passed,
        "flags": flags,
        "ans_concepts": list(ans_concepts.values()),
        "cited_chunk": f"{cited_chunk.get('source_file')} §{cited_chunk.get('section')}" if cited_chunk else None
    }


def verify_and_guard_data_claims(answer: str, tool_data_results: List[Any]) -> str:
    """
    GENERIC DATA CLAIM GUARD:
    For any answer built with invoice tool results, verifies statements that assert:
    - a count next to 'unpaid' must equal Pending+Overdue
    - next to 'overdue'/'pending'/'paid' must equal that status count
    - a total must equal total_amount
    - amounts must match a row or total
    If a statement contradicts the data, replaces it deterministically with the pure-data
    template sentence (count, total, status breakdown) with zero LLM calls.
    """
    if not tool_data_results or not answer or "I couldn't find that in the documents" in answer:
        return answer

    inv_results = [
        t for t in tool_data_results
        if isinstance(t, dict) and ("invoices" in t or "count" in t or "status_breakdown" in t)
    ]
    if not inv_results:
        return answer

    inv_res = inv_results[0]
    if not inv_res.get("found", True):
        return answer

    invoices = inv_res.get("invoices", [])
    gt_count = inv_res.get("count", len(invoices))
    gt_total = float(inv_res.get("total_amount", sum(float(i.get("amount_inr", 0)) for i in invoices)))
    sb = inv_res.get("status_breakdown", {})
    if not sb and invoices:
        sb = {}
        for inv in invoices:
            st = inv.get("status", "Unknown")
            sb[st] = sb.get(st, 0) + 1

    gt_overdue = sb.get("Overdue", 0)
    gt_pending = sb.get("Pending", 0)
    gt_paid = sb.get("Paid", 0)
    gt_unpaid = gt_overdue + gt_pending

    # Generate deterministic pure-data sentence
    from app.utils.formatters import format_inr
    sb_parts = [f"{st}: {c}" for st, c in sorted(sb.items())]
    sb_str = ", ".join(sb_parts) if sb_parts else "None"
    total_inr = format_inr(gt_total)
    crore_val = gt_total / 1e7
    lakh_val = gt_total / 1e5
    crore_str = f"{crore_val:.2f} crore"
    lakh_str = f"{lakh_val:.2f} lakh"
    unit_str = f" ({crore_str})" if gt_total >= 1e7 else (f" ({lakh_str})" if gt_total >= 1e5 else "")
    det_data_sentence = f"There are {gt_count} matching invoices with a total value of {total_inr}{unit_str}. Status breakdown: {sb_str}."

    # Separate citations suffix if present
    cits_match = re.search(r"(\n\n\[[a-zA-Z0-9_\-\.]+\.pdf[^\]]*\]\s*)+$", answer)
    cits_suffix = ""
    text_to_check = answer
    if cits_match:
        cits_suffix = answer[cits_match.start():]
        text_to_check = answer[:cits_match.start()].strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_to_check) if s.strip()]
    if not sentences:
        sentences = [text_to_check]

    valid_amounts = {float(inv.get("amount_inr", 0)) for inv in invoices}
    valid_amounts.add(gt_total)

    modified = False
    new_sentences = []
    data_sentence_inserted = False

    for sent in sentences:
        sent_lower = sent.lower()
        has_contradiction = False

        # 1. Count next to "unpaid"
        m_unpaid = re.search(r"\b(\d+)\s+(?:matching\s+)?(?:invoices?\s+(?:are\s+)?)?unpaid\b", sent_lower)
        if not m_unpaid:
            m_unpaid = re.search(r"\bunpaid\s+(?:invoices?\s*)?(?:are\s*)?(\d+)\b", sent_lower)
        if not m_unpaid:
            m_unpaid = re.search(r"\bthere\s+are\s+(\d+)\s+unpaid\b", sent_lower)
        if m_unpaid:
            claimed_unpaid = int(m_unpaid.group(1))
            if claimed_unpaid != gt_unpaid:
                has_contradiction = True

        # 2. Count next to "overdue"
        m_overdue = re.search(r"\b(\d+)\s+(?:invoices?\s+(?:are\s+)?)?overdue\b", sent_lower)
        if not m_overdue:
            m_overdue = re.search(r"\boverdue\s*:\s*(\d+)\b", sent_lower)
        if not m_overdue:
            m_overdue = re.search(r"\b(\d+)\s+are\s+overdue\b", sent_lower)
        if not m_overdue:
            m_overdue = re.search(r"\bmarked\s+as\s+['\"]?overdue['\"]?[^.\n]*?(\d+)\b", sent_lower)
        if m_overdue:
            claimed_overdue = int(m_overdue.group(1))
            if claimed_overdue != gt_overdue:
                has_contradiction = True

        # 3. Count next to "pending"
        m_pending = re.search(r"\b(\d+)\s+(?:invoices?\s+(?:are\s+)?)?pending\b", sent_lower)
        if not m_pending:
            m_pending = re.search(r"\bpending\s*:\s*(\d+)\b", sent_lower)
        if not m_pending:
            m_pending = re.search(r"\b(\d+)\s+are\s+pending\b", sent_lower)
        if m_pending:
            claimed_pending = int(m_pending.group(1))
            if claimed_pending != gt_pending:
                has_contradiction = True

        # 4. Count next to "paid"
        m_paid = re.search(r"\b(\d+)\s+(?:invoices?\s+(?:are\s+)?)?paid\b(?! date)", sent_lower)
        if not m_paid:
            m_paid = re.search(r"\bpaid\s*:\s*(\d+)\b", sent_lower)
        if m_paid:
            claimed_paid = int(m_paid.group(1))
            if claimed_paid != gt_paid:
                has_contradiction = True

        # 5. Total amount
        m_total = re.search(r"\btotal\s+(?:value|amount)?\s*(?:of)?\s*₹?\s*([0-9,]+(?:\.\d+)?)\b", sent_lower)
        if not m_total:
            m_total = re.search(r"\btotaling\s*₹?\s*([0-9,]+(?:\.\d+)?)\b", sent_lower)
        if m_total:
            clean_num = float(m_total.group(1).replace(",", ""))
            if abs(clean_num - gt_total) > 1.0:
                has_contradiction = True

        # 6. Specific invoice amount matching
        if invoices:
            m_amts = re.findall(r"₹\s*([0-9,]+(?:\.\d+)?)", sent)
            for a_str in m_amts:
                val = float(a_str.replace(",", ""))
                if not any(abs(val - v) < 1.0 for v in valid_amounts):
                    if any(w in sent_lower for w in ["invoice", "total", "amount", "value", "overdue", "pending"]):
                        has_contradiction = True

        if has_contradiction:
            modified = True
            if not data_sentence_inserted:
                new_sentences.append(det_data_sentence)
                data_sentence_inserted = True
        else:
            new_sentences.append(sent)

    if not modified:
        return answer

    res = " ".join(new_sentences)
    if cits_suffix:
        res = res.strip() + cits_suffix
    return res
