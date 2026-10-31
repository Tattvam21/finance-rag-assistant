import os
import re
import json
import time
import string
import difflib
import logging
from typing import Dict, Any, List, Optional, Tuple

from app.config import (
    OLLAMA_HOST,
    CHAT_MODEL,
    GATE_MODEL,
    NUM_CTX,
    SIMILARITY_THRESHOLD,
    OLLAMA_TIMEOUT,
    get_ollama_client
)
from app.llm.schemas import ChatResponse, InvoiceRow, InvoiceAnswer, QueryPlan, InvoiceFilter, StructuredAnswer
from app.llm.prompts import SYSTEM_PROMPT, FAITHFULNESS_PROMPT, ANSWERABILITY_GATE_PROMPT, QUERY_PLANNER_PROMPT, STRUCTURED_ANSWER_PROMPT
from app.retrieval.search import search_chunks_with_timing
from app.tools.invoice_queries import query_invoices, get_distinct_vendors
from app.tools.late_fee import calculate_late_fee, format_late_fee_template
from app.utils.formatters import format_inr, format_inr_lakh_crore
from app.utils.numeric_grounding import build_verified_numeric_facts

logger = logging.getLogger("finance_rag.generate")

# -------------------------------------------------------------
# Ollama Tool Definitions (Tasks 1, 4, 7)
# -------------------------------------------------------------
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": "Searches corporate finance policy documents for payment terms, late fees, SLAs, travel/food per diem allowances, TDS withholding tax rates, GST compliance, and dispute resolution guidelines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific policy topic or keywords to search for in corporate guidelines."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_invoices",
            "description": "Queries company invoices with parameterized filters. For unpaid invoices, use status='unpaid' (covers Pending and Overdue). Supports exact invoice_id lookup (e.g. 'INV-2024-001'), minimum/maximum amounts in INR, and vendor names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "Exact invoice ID to look up (e.g. 'INV-2024-001')."
                    },
                    "status": {
                        "type": "string",
                        "description": "Status filter: 'unpaid' (Pending & Overdue), 'Paid', 'Pending', 'Overdue', 'Disputed', 'Partially Paid'."
                    },
                    "min_amount": {
                        "type": "number",
                        "description": "Minimum invoice amount in INR (e.g., 100000 for 1 lakh)."
                    },
                    "max_amount": {
                        "type": "number",
                        "description": "Maximum invoice amount in INR."
                    },
                    "vendor": {
                        "type": "string",
                        "description": "Vendor name or partial substring."
                    },
                    "due_before": {
                        "type": "string",
                        "description": "Due date limit in YYYY-MM-DD format."
                    },
                    "department": {
                        "type": "string",
                        "description": "Department filter (IT, Operations, Finance, HR, Legal)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_late_fee",
            "description": "Calculates late interest fees for an invoice according to corporate policy rate on principal for days overdue past due_date. Always call this tool for fee calculation questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {
                        "type": "string",
                        "description": "The invoice ID to calculate late fees for (e.g. 'INV-2024-036')."
                    },
                    "as_of_date": {
                        "type": "string",
                        "description": "Optional calculation date in YYYY-MM-DD format (defaults to current date)."
                    }
                },
                "required": ["invoice_id"]
            }
        }
    }
]

def is_pure_greeting(text: str) -> bool:
    """Detects simple conversational greetings and meta-prompts that do not require tool retrieval."""
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.strip().lower())
    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "what can you do", "help", "who are you"}
    return clean in greetings or len(clean.split()) <= 1 and clean in {"hi", "hello", "hey"}

MAX_PROMPT_TOKENS_SEEN = 0

def estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimates tokens in messages payload: roughly 3.5 characters per token + framing overhead."""
    total_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            total_chars += len(json.dumps(c))
        tc = m.get("tool_calls")
        if tc:
            total_chars += len(json.dumps(tc))
    return int(total_chars / 3.5) + len(messages) * 4

LAST_CHAT_PROMPT_TOKENS = 0

def get_max_prompt_tokens() -> int:
    global MAX_PROMPT_TOKENS_SEEN
    return MAX_PROMPT_TOKENS_SEEN

def get_last_chat_prompt_tokens() -> int:
    global LAST_CHAT_PROMPT_TOKENS
    return LAST_CHAT_PROMPT_TOKENS

def execute_chat_with_timeout(client, **kwargs):
    """Executes client.chat with timeout, 1 retry, and prompt token size guards."""
    global MAX_PROMPT_TOKENS_SEEN, LAST_CHAT_PROMPT_TOKENS
    messages = kwargs.get("messages", [])
    prompt_tokens = estimate_prompt_tokens(messages)
    LAST_CHAT_PROMPT_TOKENS = prompt_tokens
    if prompt_tokens > MAX_PROMPT_TOKENS_SEEN:
        MAX_PROMPT_TOKENS_SEEN = prompt_tokens

    logger.info(f"Ollama chat call prompt tokens: ~{prompt_tokens} (NUM_CTX: {NUM_CTX})")

    fail_limit = int(0.95 * NUM_CTX)
    warn_limit = int(0.80 * NUM_CTX)

    if prompt_tokens > fail_limit:
        raise ValueError(f"Prompt size {prompt_tokens} tokens exceeds 95% of NUM_CTX ({NUM_CTX}) limit ({fail_limit})!")
    elif prompt_tokens > warn_limit:
        logger.warning(f"Warning: Prompt size {prompt_tokens} tokens exceeds 80% of NUM_CTX ({NUM_CTX}) threshold ({warn_limit}).")

    for attempt in range(2):
        try:
            return client.chat(**kwargs)
        except Exception as e:
            if attempt == 1:
                raise e
            logger.warning(f"Ollama chat attempt 1 failed ({e}). Retrying...")
            time.sleep(1.0)

def verify_answer_faithfulness(answer: str, chunks: List[Dict[str, Any]], model: str = CHAT_MODEL) -> Dict[str, Any]:
    """Runs post-generation audit evaluating whether all claims are supported by context."""
    client = get_ollama_client()
    context_str = "\n\n".join([f"[{c['source_file']}, p.{c['page']}, §{c['section']}]:\n{c['content']}" for c in chunks])
    prompt = FAITHFULNESS_PROMPT.format(context=context_str, answer=answer)

    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format={"type": "object", "properties": {"supported": {"type": "boolean"}, "unsupported_claims": {"type": "array", "items": {"type": "string"}}}, "required": ["supported"]},
            options={"temperature": 0.0, "num_ctx": NUM_CTX},
            think=False
        )
        return json.loads(resp.message.content)
    except Exception as e:
        logger.warning(f"Faithfulness verification check error: {e}")
        return {"supported": True, "unsupported_claims": []}

def normalize_for_quote_match(text: str) -> str:
    """Lowers, strips punctuation, and collapses whitespace."""
    text = text.lower()
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def verify_quote_against_chunks(quote: str, chunks: List[Dict[str, Any]]) -> Tuple[bool, Optional[int]]:
    """
    Accepts the quote if it is a substring of a chunk OR matches a chunk sentence with difflib ratio >= 0.9.
    Returns (verified, matching_chunk_index).
    """
    norm_q = normalize_for_quote_match(quote)
    if not norm_q:
        return False, None

    # 1. Direct substring match against chunk content
    for idx, c in enumerate(chunks):
        norm_c = normalize_for_quote_match(c.get("content", ""))
        if norm_q in norm_c:
            return True, idx

    # 2. Sentence-level matching (substring or difflib ratio >= 0.9)
    for idx, c in enumerate(chunks):
        content = c.get("content", "")
        sentences = [s.strip() for s in re.split(r'(?<!\d)[.!?\n]+(?!\d)', content) if s.strip()]
        for sent in sentences:
            norm_sent = normalize_for_quote_match(sent)
            if not norm_sent:
                continue
            if norm_q in norm_sent or norm_sent in norm_q:
                return True, idx
            ratio = difflib.SequenceMatcher(None, norm_q, norm_sent).ratio()
            if ratio >= 0.9:
                return True, idx

    return False, None

def check_answerability_gate(
    question: str,
    chunks: List[Dict[str, Any]],
    tool_results: Optional[List[Any]] = None,
    gate_model: str = GATE_MODEL,
    debug: bool = False
) -> Tuple[bool, str, Optional[int]]:
    """
    Checks if retrieved chunks contain direct factual support for the question.
    Gate calls: think=False, num_predict <= 150, temperature 0.
    Verifies quote: lowercase, collapse whitespace, strip punctuation, substring or sentence difflib >= 0.9.
    If quote verification fails, retries gate once with 'copy the sentence exactly'.
    Returns (answerable, quote, matching_chunk_index).
    """
    if not chunks:
        return False, "", None
    client = get_ollama_client()
    context_blocks = [f"[{c['source_file']}, p.{c['page']}, §{c['section']}]:\n{c['content']}" for c in chunks]
    if tool_results:
        tool_payloads = [t.get("model_payload", t) if isinstance(t, dict) else t for t in tool_results]
        context_blocks.append(f"[Invoice Database Context]:\n{json.dumps(tool_payloads, default=str)}")
    context_str = "\n\n".join(context_blocks)
    numeric_facts, matching_quote = build_verified_numeric_facts(question, chunks, tool_results)
    num_facts_block = f"\n{numeric_facts}\n" if numeric_facts else ""
    prompt = ANSWERABILITY_GATE_PROMPT.format(question=question, context=context_str, numeric_facts=num_facts_block)

    try:
        t0 = time.time()
        resp = execute_chat_with_timeout(
            client=client,
            model=gate_model,
            messages=[{"role": "user", "content": prompt}],
            format={"type": "object", "properties": {"answerable": {"type": "boolean"}, "supporting_quote": {"type": "string"}}, "required": ["answerable", "supporting_quote"]},
            options={"temperature": 0.0, "num_ctx": NUM_CTX, "num_predict": 150},
            think=False
        )
        gate_dur = round((time.time() - t0) * 1000, 2)
        res_data = json.loads(resp.message.content)
        answerable = bool(res_data.get("answerable", False))
        quote = str(res_data.get("supporting_quote", "") or "").strip()

        # Task 2 requirement: For P04 print gate JSON and §1.2 chunk text
        if "bulk disbursements" in question.lower():
            if debug:
                print(f"[DEBUG] P04 Gate JSON: {resp.message.content}")
                for c in chunks:
                    if "1.2" in str(c.get("section", "")):
                        print(f"[DEBUG] P04 §1.2 Chunk Text:\n{c.get('content')}")

        quote_verified, chunk_idx = verify_quote_against_chunks(quote, chunks)

        # If a verified numeric matching quote exists and quote verification failed or quote was empty
        if answerable and (not quote_verified or matching_quote):
            if matching_quote:
                m_ver, m_idx = verify_quote_against_chunks(matching_quote, chunks)
                if m_ver:
                    quote = matching_quote
                    quote_verified = True
                    chunk_idx = m_idx

        # If quote fails verification, retry gate once with "copy the sentence exactly"
        if answerable and not quote_verified:
            if debug:
                print(f"[DEBUG] Gate quote '{quote[:50]}...' failed verification. Retrying gate once with exact copy request...")
            retry_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resp.message.content},
                {"role": "user", "content": "The supporting_quote could not be verified in the documents. Copy the sentence exactly from the context."}
            ]
            t_retry = time.time()
            retry_resp = execute_chat_with_timeout(
                client=client,
                model=gate_model,
                messages=retry_messages,
                format={"type": "object", "properties": {"answerable": {"type": "boolean"}, "supporting_quote": {"type": "string"}}, "required": ["answerable", "supporting_quote"]},
                options={"temperature": 0.0, "num_ctx": NUM_CTX, "num_predict": 150},
                think=False
            )
            retry_data = json.loads(retry_resp.message.content)
            answerable = bool(retry_data.get("answerable", False))
            quote = str(retry_data.get("supporting_quote", "") or "").strip()
            quote_verified, chunk_idx = verify_quote_against_chunks(quote, chunks)
            if debug:
                print(f"[DEBUG] Gate retry finished in {round((time.time() - t_retry)*1000, 2)}ms. Verified={quote_verified}")

        if debug:
            status_str = "PASSED" if (answerable and quote_verified) else "REFUSED"
            print(f"[DEBUG] Answer Gate ({gate_dur}ms) [{status_str}]: Answerable={answerable} | Quote Verified={quote_verified} | Quote='{quote[:60]}...'")

        if answerable and quote_verified:
            return True, quote, chunk_idx
        return False, quote, None
    except Exception as e:
        logger.warning(f"Answerability gate error: {e}")
        if debug:
            print(f"[DEBUG] Answer Gate Error: {e}")
        return False, "", None

def check_numeric_grounding(
    answer: str,
    tool_results: List[Any],
    chunks: List[Dict[str, Any]],
    question: str
) -> Tuple[bool, List[str]]:
    """
    Verifies that every ₹ amount and every number with 3+ digits in the final answer
    must appear (after removing commas and the ₹ sign) in the tool results, retrieved chunks, or question.
    """
    context_parts = [question]
    for tr in tool_results:
        if isinstance(tr, (dict, list)):
            context_parts.append(json.dumps(tr, default=str))
        else:
            context_parts.append(str(tr))
    for c in chunks:
        context_parts.append(c.get("content", ""))
        context_parts.append(f"{c.get('source_file', '')} p.{c.get('page', '')} §{c.get('section', '')}")

    combined_context = " ".join(context_parts)
    norm_ctx = combined_context.replace("₹", "").replace(",", "")

    # Extract ₹ amounts
    rupee_matches = re.findall(r'₹\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)', answer)
    # Extract numbers with 3 or more digits
    big_num_matches = re.findall(r'\b([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?|[0-9]{3,}(?:\.[0-9]+)?)\b', answer)

    all_raw = list(dict.fromkeys(rupee_matches + big_num_matches))
    offending: List[str] = []

    for raw in all_raw:
        clean_num = raw.replace("₹", "").replace(",", "").strip()
        if not clean_num:
            continue
        if clean_num in norm_ctx:
            continue
        # Check integer representation if ending in .00
        if clean_num.endswith(".00") and clean_num[:-3] in norm_ctx:
            continue
        offending.append(clean_num)

    return (len(offending) == 0, offending)

def extract_citations(text: str) -> List[str]:
    """Parses citations matching [file.pdf, p.X, §Y] or [file.pdf, §Y] or [X]."""
    matches = re.findall(r'\[([a-zA-Z0-9_\-\.]+\.pdf(?:,\s*p\.\d+)?(?:,\s*§[^\n\]]+)?)\]', text)
    return list(dict.fromkeys(matches))

def append_condition_to_answer(answer: str, condition_sentence: str, cited_chunk: Optional[Dict[str, Any]] = None) -> str:
    """
    Deterministically appends a condition sentence verbatim to the answer
    preserving existing citations and punctuation, with zero LLM calls.
    """
    if not condition_sentence:
        return answer
    cond_clean = condition_sentence.strip()
    if not cond_clean.endswith("."):
        cond_clean += "."

    # Avoid duplicate appending if words are already in answer
    cond_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", cond_clean) if w.lower() not in ("that", "this", "with", "from", "for", "will")]
    overlap = [w for w in cond_words if w in answer.lower()]
    if len(overlap) >= min(3, len(cond_words)):
        return answer

    # Case 1: Answer has block citations at the end (\n\n[file.pdf, ...])
    m_block = re.search(r'(\n\n\s*\[[a-zA-Z0-9_\-\.]+\.pdf[^\]]*\](?:.*))$', answer, flags=re.DOTALL)
    if m_block:
        body = answer[:m_block.start()].rstrip()
        cits = m_block.group(1)
        if not body.endswith((".", "!", "?")):
            body += "."
        return f"{body} {cond_clean}{cits}"

    # Case 2: Answer has inline citation at the end ([file.pdf, ...])
    m_inline = re.search(r'(\s*\[[a-zA-Z0-9_\-\.]+\.pdf[^\]]*\][\.\s]*)$', answer)
    if m_inline:
        body = answer[:m_inline.start()].rstrip()
        cit = m_inline.group(1).strip()
        if not body.endswith((".", "!", "?")):
            body += "."
        return f"{body} {cond_clean} {cit}"

    # Case 3: No citation found in answer
    body = answer.rstrip()
    if not body.endswith((".", "!", "?")):
        body += "."
    cit_tag = ""
    if cited_chunk:
        cit_tag = f"\n\n[{cited_chunk.get('source_file')}, p.{cited_chunk.get('page', 1)}, §{cited_chunk.get('section', '')}]"
    return f"{body} {cond_clean}{cit_tag}"


POLICY_KEYWORDS = {
    "policy", "rule", "guideline", "escalation", "phase", "method",
    "term", "terms", "condition", "sla", "allowance", "per diem",
    "tds", "gst", "travel", "disbursement", "rate", "grace period",
    "cap", "holding", "matching", "dispute", "blacklist",
    "reimbursement", "retention", "statutory", "audit", "flight", "hotel"
}

CANONICAL_INV_RE = re.compile(r"\b(INV-\d{4}-\d{3})\b", re.IGNORECASE)

NON_CANONICAL_PATTERNS = [
    re.compile(r"\b(INV\s*[-_ ]?\s*0*(\d{1,4}))\b", re.IGNORECASE),
    re.compile(r"\b(invoice\s+(?:number\s+|no\.?\s*|#\s*)?0*(\d{1,4}))\b", re.IGNORECASE),
    re.compile(r"(?:^|\s)(#0*(\d{1,4}))\b", re.IGNORECASE),
]

def parse_invoice_reference(question: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parses invoice reference from question.
    Returns: (canonical_id, non_canonical_raw_text, suggested_canonical_id)
    - If user wrote canonical form (case-insensitive INV-\\d{4}-\\d{3}), returns (CANONICAL_ID, None, None).
    - If user wrote non-canonical form (INV001, INV 036, invoice number 36, #36), returns (None, raw_text, suggested_id).
    - If no invoice reference, returns (None, None, None).
    """
    m_canon = CANONICAL_INV_RE.search(question)
    if m_canon:
        return m_canon.group(1).upper(), None, None
    for pat in NON_CANONICAL_PATTERNS:
        m = pat.search(question)
        if m:
            raw_text = m.group(1).strip()
            num = int(m.group(2))
            suggested = f"INV-2024-{num:03d}"
            return None, raw_text, suggested
    return None, None, None

def detect_question_entities(question: str, backend: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministically extracts invoice IDs, vendor names, late fee intents,
    and identifies pure data questions.
    """
    q_lower = question.lower()
    # 1. Invoice ID: only canonical invoice IDs are extracted for querying
    canon_id, non_canon_raw, _ = parse_invoice_reference(question)
    invoice_id = canon_id

    # 2. Vendor name from DISTINCT vendors
    matched_vendor = None
    try:
        vendors = get_distinct_vendors(backend=backend)
        vendors_sorted = sorted(vendors, key=len, reverse=True)
        for v in vendors_sorted:
            if v.lower() in q_lower:
                matched_vendor = v
                break
    except Exception as e:
        logger.warning(f"Error fetching distinct vendors: {e}")

    # 3. Late fee amount check
    asks_late_fee = "late fee" in q_lower
    asks_amount = any(w in q_lower for w in ["amount", "how much", "owed", "calculate", "total due", "fee is owed"]) and "rate" not in q_lower
    is_late_fee_amount = bool(invoice_id and asks_late_fee and asks_amount)

    # 4. Pure data question detection:
    # only an invoice ID/vendor and words like status/overdue/paid/pending/total/show
    has_policy_kw = any(kw in q_lower for kw in POLICY_KEYWORDS)
    is_pure_data = False
    if not has_policy_kw:
        data_words = ["status", "overdue", "paid", "pending", "unpaid", "total", "show", "list", "highest", "how many", "invoices", "invoice"]
        has_data_indicator = any(dw in q_lower for dw in data_words)
        if (invoice_id or matched_vendor) and has_data_indicator and not is_late_fee_amount:
            is_pure_data = True
        elif is_late_fee_amount and not has_policy_kw:
            is_pure_data = True

    return {
        "invoice_id": invoice_id,
        "vendor": matched_vendor,
        "is_late_fee_amount": is_late_fee_amount,
        "is_pure_data": is_pure_data
    }

def _chat_legacy_tools(
    question: str,
    model: Optional[str] = None,
    router: str = "llm",
    backend: Optional[str] = None,
    retrieval_strategy: str = "hybrid",
    retrieval_mode: str = "always",
    keyword_bypass: bool = False,
    answer_gate: bool = True,
    gate_model: Optional[str] = None,
    use_reranker: bool = False,
    verify: bool = False,
    debug: bool = False
) -> ChatResponse:
    """
    Main Assistant chat handler.
    - Deterministically detects entities and executes forced tools by code.
    - If pure data question, skips policy chunks to eliminate noise.
    - Runs Ollama tool-calling loop.
    - Runs Answerability Gate AFTER tools. Refuses only if not data_ok, not not_found_ok, and gate says unsupported.
    - Appends verified policy citations in code. Suppresses policy citations for pure invoice answers.
    - Enforces numeric grounding for all answers.
    - Tracks stage timings and identifies slowest stage.
    """
    start_time = time.time()
    target_model = model or CHAT_MODEL
    client = get_ollama_client()

    t_stage_retrieval = 0.0
    t_stage_forced_tools = 0.0
    t_stage_chat_calls = 0.0
    t_stage_gate = 0.0
    t_stage_cite_retry = 0.0
    t_stage_ground_retry = 0.0
    chat_durations: List[float] = []

    all_invoices: List[Dict[str, Any]] = []
    executed_tools: List[Dict[str, Any]] = []
    actual_retrieved_chunks: List[Dict[str, Any]] = []
    queries_run: List[str] = []
    latest_fee_data: Optional[Dict[str, Any]] = None
    tool_data_results: List[Any] = []

    # Detect entities deterministically in code
    entities = detect_question_entities(question, backend=backend)
    is_pure_data = entities["is_pure_data"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    final_answer = ""
    tool_calls_executed = 0

    # Task 3: In "always" mode, search_policies runs UNLESS it's a pure data question or greeting
    if retrieval_mode == "always":
        if is_pure_data:
            if debug:
                print(f"[DEBUG] Pure data question detected: skipping policy chunk retrieval.")
        elif not is_pure_greeting(question):
            if debug:
                print(f"[DEBUG] Always-retrieve mode: pre-fetching policy chunks for query '{question}'.")
                print(f"[DEBUG] Tool Call: search_policies({json.dumps({'query': question})})")
            executed_tools.append({"name": "search_policies", "args": {"query": question}, "forced": False})
            t0_ret = time.time()
            chunks, emb_ms, ret_ms = search_chunks_with_timing(
                query=question,
                limit=5,
                retrieval_mode=retrieval_strategy,
                use_reranker=use_reranker,
                backend=backend
            )
            t_stage_retrieval += (time.time() - t0_ret) * 1000
            actual_retrieved_chunks.extend(chunks)
            queries_run.append(question)
            tool_calls_executed += 1

            best_sim = max((c.get("similarity", 0.0) for c in chunks), default=0.0)
            has_keyword_match = any(c.get("keyword_rank") is not None for c in chunks[:3])
            allow_bypass = keyword_bypass and has_keyword_match
            if not chunks or (best_sim < SIMILARITY_THRESHOLD and not (retrieval_strategy == "hybrid" and allow_bypass)):
                tool_content = "I couldn't find that in the documents."
            else:
                context_blocks = [
                    f"[{c['source_file']}, p.{c['page']}, §{c['section']}]:\n{c['content']}"
                    for c in chunks
                ]
                tool_content = "\n\n".join(context_blocks)

            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": "always_retrieval_call",
                    "function": {"name": "search_policies", "arguments": {"query": question}}
                }]
            })
            messages.append({
                "role": "tool",
                "content": tool_content
            })

    # Task 3: FORCED TOOLS BY ENTITY (deterministic, in code)
    if entities["invoice_id"]:
        inv_id = entities["invoice_id"]
        t0_f = time.time()
        inv_res = query_invoices(invoice_id=inv_id, backend=backend)
        t_stage_forced_tools += (time.time() - t0_f) * 1000
        tool_data_results.append(inv_res)
        if inv_res.get("invoices"):
            all_invoices.extend(inv_res["invoices"])
        executed_tools.append({"name": "query_invoices", "args": {"invoice_id": inv_id}, "forced": True})
        tool_calls_executed += 1
        if debug:
            print(f"[DEBUG] forced tool: query_invoices (invoice ID detected: {inv_id})")
            print(f"[DEBUG] Tool Call: query_invoices({json.dumps({'invoice_id': inv_id})})")
        payload = inv_res.get("model_payload", inv_res)
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": "forced_query_invoices",
                "function": {"name": "query_invoices", "arguments": {"invoice_id": inv_id}}
            }]
        })
        messages.append({
            "role": "tool",
            "content": json.dumps(payload, default=str)
        })

    if entities["is_late_fee_amount"]:
        inv_id = entities["invoice_id"]
        t0_f = time.time()
        fee_res = calculate_late_fee(invoice_id=inv_id, backend=backend)
        t_stage_forced_tools += (time.time() - t0_f) * 1000
        latest_fee_data = fee_res
        tool_data_results.append(fee_res)
        executed_tools.append({"name": "calculate_late_fee", "args": {"invoice_id": inv_id}, "forced": True})
        tool_calls_executed += 1
        if debug:
            print(f"[DEBUG] forced tool: calculate_late_fee (late fee amount requested for {inv_id})")
            print(f"[DEBUG] Tool Call: calculate_late_fee({json.dumps({'invoice_id': inv_id})})")
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": "forced_late_fee",
                "function": {"name": "calculate_late_fee", "arguments": {"invoice_id": inv_id}}
            }]
        })
        messages.append({
            "role": "tool",
            "content": json.dumps(fee_res, default=str)
        })

    elif entities["vendor"]:
        v_name = entities["vendor"]
        q_low = question.lower()
        st = None
        if "pending" in q_low:
            st = "pending"
        elif "overdue" in q_low:
            st = "overdue"
        elif "unpaid" in q_low:
            st = "unpaid"
        elif "paid" in q_low:
            st = "paid"
        v_args: Dict[str, Any] = {"vendor": v_name}
        if st:
            v_args["status"] = st

        t0_f = time.time()
        inv_res = query_invoices(backend=backend, **v_args)
        t_stage_forced_tools += (time.time() - t0_f) * 1000
        tool_data_results.append(inv_res)
        if inv_res.get("invoices"):
            all_invoices.extend(inv_res["invoices"])
        executed_tools.append({"name": "query_invoices", "args": v_args, "forced": True})
        tool_calls_executed += 1
        if debug:
            print(f"[DEBUG] forced tool: query_invoices (vendor detected: {v_name})")
            print(f"[DEBUG] Tool Call: query_invoices({json.dumps(v_args)})")
        payload = inv_res.get("model_payload", inv_res)
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": "forced_vendor_query",
                "function": {"name": "query_invoices", "arguments": v_args}
            }]
        })
        messages.append({
            "role": "tool",
            "content": json.dumps(payload, default=str)
        })

    # Task 1 check: If calculate_late_fee was forced for a pure late fee query, render template immediately
    if entities["is_late_fee_amount"] and latest_fee_data and latest_fee_data.get("found") and is_pure_data:
        final_answer = format_late_fee_template(latest_fee_data)
        if debug:
            print("[DEBUG] Rendered calculate_late_fee answer via deterministic Python template.")

    # Main iterative tool loop (max 4 steps)
    if not final_answer:
        for iteration in range(4):
            prompt_chars = sum(len(m.get("content", "") or "") for m in messages)
            prompt_tokens_est = prompt_chars // 4
            if debug:
                print(f"[DEBUG] --- Ollama Tool Loop Step {iteration + 1}/4 --- (Prompt size: ~{prompt_tokens_est} tokens)")

            t_chat_start = time.time()
            response = execute_chat_with_timeout(
                client=client,
                model=target_model,
                messages=messages,
                tools=OLLAMA_TOOLS,
                options={"temperature": 0.0, "num_ctx": NUM_CTX},
                think=False,
                keep_alive="30m"
            )
            chat_dur = round((time.time() - t_chat_start) * 1000, 2)
            t_stage_chat_calls += chat_dur
            chat_durations.append(chat_dur)
            if debug:
                eval_count = getattr(response, "eval_count", None)
                eval_duration = getattr(response, "eval_duration", None)
                tps_str = ""
                if eval_count and eval_duration:
                    tps = round(eval_count / (eval_duration / 1e9), 1)
                    tps_str = f" | {tps} tokens/sec"
                print(f"[DEBUG] ollama.chat call duration: {chat_dur}ms{tps_str}")

            msg = response.message
            tool_calls = msg.tool_calls

            if not tool_calls:
                # If model produced final answer with 0 tool calls on a financial query and no forced tools ran
                if tool_calls_executed == 0 and not is_pure_greeting(question) and not is_pure_data:
                    if debug:
                        print(f"[DEBUG] Forced retrieval triggered: zero tool calls on financial query '{question}'.")
                        print(f"[DEBUG] Tool Call: search_policies({json.dumps({'query': question})})")
                    executed_tools.append({"name": "search_policies", "args": {"query": question}, "forced": False})
                    t0_ret = time.time()
                    chunks, emb_ms, ret_ms = search_chunks_with_timing(
                        query=question,
                        limit=5,
                        retrieval_mode=retrieval_strategy,
                        use_reranker=use_reranker,
                        backend=backend
                    )
                    t_stage_retrieval += (time.time() - t0_ret) * 1000
                    actual_retrieved_chunks.extend(chunks)
                    queries_run.append(question)
                    tool_calls_executed += 1

                    best_sim = max((c.get("similarity", 0.0) for c in chunks), default=0.0)
                    has_keyword_match = any(c.get("keyword_rank") is not None for c in chunks[:3])
                    allow_bypass = keyword_bypass and has_keyword_match
                    if not chunks or (best_sim < SIMILARITY_THRESHOLD and not (retrieval_strategy == "hybrid" and allow_bypass)):
                        final_answer = "I couldn't find that in the documents."
                        break

                    context_blocks = [
                        f"[{c['source_file']}, p.{c['page']}, §{c['section']}]:\n{c['content']}"
                        for c in chunks
                    ]
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "forced_call",
                            "function": {"name": "search_policies", "arguments": {"query": question}}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "content": "\n\n".join(context_blocks)
                    })
                    continue

                final_answer = msg.content or ""
                break

            messages.append(msg)

            for tc in tool_calls:
                tool_calls_executed += 1
                fn_name = tc.function.name
                fn_args = tc.function.arguments or {}
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {}

                if debug:
                    print(f"[DEBUG] Model-initiated tool call: {fn_name}({json.dumps(fn_args)})")
                    print(f"[DEBUG] Tool Call: {fn_name}({json.dumps(fn_args)})")
                executed_tools.append({"name": fn_name, "args": fn_args, "forced": False})

                if fn_name == "search_policies":
                    q = fn_args.get("query", question)
                    queries_run.append(q)
                    t0_ret = time.time()
                    chunks, emb_ms, ret_ms = search_chunks_with_timing(
                        query=q,
                        limit=5,
                        retrieval_mode=retrieval_strategy,
                        use_reranker=use_reranker,
                        backend=backend
                    )
                    t_stage_retrieval += (time.time() - t0_ret) * 1000
                    actual_retrieved_chunks.extend(chunks)

                    best_sim = max((c.get("similarity", 0.0) for c in chunks), default=0.0)
                    has_keyword_match = any(c.get("keyword_rank") is not None for c in chunks[:3])
                    allow_bypass = keyword_bypass and has_keyword_match
                    if not chunks or (best_sim < SIMILARITY_THRESHOLD and not (retrieval_strategy == "hybrid" and allow_bypass)):
                        tool_content = "I couldn't find that in the documents."
                    else:
                        context_blocks = [
                            f"[{c['source_file']}, p.{c['page']}, §{c['section']}]:\n{c['content']}"
                            for c in chunks
                        ]
                        tool_content = "\n\n".join(context_blocks)

                    messages.append({
                        "role": "tool",
                        "content": tool_content
                    })

                elif fn_name == "query_invoices":
                    t0_inv = time.time()
                    inv_res = query_invoices(backend=backend, **fn_args)
                    t_stage_forced_tools += (time.time() - t0_inv) * 1000
                    tool_data_results.append(inv_res)
                    if inv_res.get("invoices"):
                        all_invoices.extend(inv_res["invoices"])

                    payload_to_send = inv_res.get("model_payload", inv_res)
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(payload_to_send, default=str)
                    })

                elif fn_name == "calculate_late_fee":
                    t0_fee = time.time()
                    fee_res = calculate_late_fee(backend=backend, **fn_args)
                    t_stage_forced_tools += (time.time() - t0_fee) * 1000
                    latest_fee_data = fee_res
                    tool_data_results.append(fee_res)
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(fee_res, default=str)
                    })

    # Task 2: REFUSAL LOGIC AND GATE ORDER
    # data_ok = any query_invoices or calculate_late_fee result with found > 0
    # not_found_ok = query_invoices returned an explicit not-found result
    data_ok = any(
        (isinstance(t, dict) and (t.get("found") is True or t.get("count", 0) > 0 or len(t.get("invoices", [])) > 0))
        for t in tool_data_results
    )
    not_found_ok = any(
        (isinstance(t, dict) and t.get("found") is False and ("searched_id" in t or "suggested_ids" in t))
        for t in tool_data_results
    )

    gate_supported = True
    gate_quote = ""
    gate_chunk_idx: Optional[int] = None
    gate_model_target = gate_model or GATE_MODEL

    # The gate must run after the tool loop (or after forced tools), never before the model can reach the invoice tools.
    if answer_gate and actual_retrieved_chunks and not is_pure_data:
        t0_gate = time.time()
        gate_supported, gate_quote, gate_chunk_idx = check_answerability_gate(
            question=question,
            chunks=actual_retrieved_chunks,
            gate_model=gate_model_target,
            debug=debug
        )
        t_stage_gate = round((time.time() - t0_gate) * 1000, 2)

    # Final rule: refuse with "I couldn't find that in the documents." only if NOT data_ok,
    # NOT not_found_ok, AND the gate says the policy chunks do not support an answer.
    if (not data_ok) and (not not_found_ok) and (not gate_supported):
        final_answer = "I couldn't find that in the documents."
        if debug:
            print(f"[DEBUG] Final refusal triggered: data_ok={data_ok}, not_found_ok={not_found_ok}, gate_supported={gate_supported} -> '{final_answer}'")

    # If calculate_late_fee was called for a fee amount question, render deterministically via Python template
    if not final_answer and latest_fee_data and latest_fee_data.get("found"):
        if entities["is_late_fee_amount"] or ("late fee" in question.lower() and any(w in question.lower() for w in ["how much", "owed", "calculate", "what is"])):
            final_answer = format_late_fee_template(latest_fee_data)
            if debug:
                print(f"[DEBUG] Rendered calculate_late_fee answer via deterministic Python template.")

    # If final answer not yet synthesized, run final Ollama synthesis
    if not final_answer:
        t_final = time.time()
        final_resp = execute_chat_with_timeout(
            client=client,
            model=target_model,
            messages=messages,
            options={"temperature": 0.0, "num_ctx": NUM_CTX},
            think=False,
            keep_alive="30m"
        )
        synth_dur = round((time.time() - t_final) * 1000, 2)
        t_stage_chat_calls += synth_dur
        chat_durations.append(synth_dur)
        final_answer = final_resp.message.content or ""
        if debug:
            print(f"[DEBUG] Final synthesis call duration: {synth_dur}ms")

    # Task 4: CITATIONS APPENDED BY CODE
    if "I couldn't find that in the documents" not in final_answer:
        # If data_ok and the gate says not supported, answer from data only with no policy citation.
        # Pure invoice answers get no policy citation.
        if is_pure_data or (data_ok and not gate_supported) or (all_invoices and not actual_retrieved_chunks):
            final_answer = re.sub(r'\[[a-zA-Z0-9_\-\.]+\.pdf[^\]]*\]', '', final_answer).strip()
        else:
            # When the gate returns a verified supporting_quote, find chunk and append [file.pdf, p.X, §Y] in code
            if gate_supported and gate_chunk_idx is not None and gate_chunk_idx < len(actual_retrieved_chunks):
                ver_chunk = actual_retrieved_chunks[gate_chunk_idx]
                code_citation = f"[{ver_chunk['source_file']}, p.{ver_chunk['page']}, §{ver_chunk['section']}]"
                if ver_chunk['source_file'] not in final_answer:
                    final_answer = f"{final_answer}\n\n{code_citation}"

            # If the model also wrote citations, keep only those that match retrieved chunks and flag the rest as UNVERIFIED CITATION
            ret_files = {c['source_file'] for c in actual_retrieved_chunks}
            for cit in extract_citations(final_answer):
                if not any(cit.startswith(rf) or rf in cit for rf in ret_files):
                    final_answer = final_answer.replace(f"[{cit}]", f"[UNVERIFIED CITATION: {cit}]")

    # Task 1: Numeric Grounding Check for ALL answers
    if "I couldn't find that in the documents" not in final_answer:
        all_tool_outputs = [m.get("content", "") for m in messages if m.get("role") == "tool"]
        num_ok, offending = check_numeric_grounding(
            answer=final_answer,
            tool_results=all_tool_outputs,
            chunks=actual_retrieved_chunks,
            question=question
        )
        if not num_ok:
            if debug:
                print(f"[DEBUG] Numeric Grounding Check failed. Offending numbers: {offending}. Retrying once...")
            regen_prompt = (
                f"Your previous answer contained ungrounded numbers or ₹ amounts not supported by the context: {offending}. "
                "Rewrite your answer, strictly using ONLY the exact numbers present in the retrieved tool and document context."
            )
            messages.append({"role": "user", "content": regen_prompt})
            t_num = time.time()
            retry_resp = execute_chat_with_timeout(
                client=client,
                model=target_model,
                messages=messages,
                options={"temperature": 0.0, "num_ctx": NUM_CTX},
                think=False,
                keep_alive="30m"
            )
            g_dur = round((time.time() - t_num) * 1000, 2)
            t_stage_ground_retry += g_dur
            retry_ans = retry_resp.message.content or ""
            retry_ok, retry_offending = check_numeric_grounding(
                answer=retry_ans,
                tool_results=all_tool_outputs,
                chunks=actual_retrieved_chunks,
                question=question
            )
            if retry_ok:
                final_answer = retry_ans
                if debug:
                    print(f"[DEBUG] Numeric Grounding retry passed in {g_dur}ms.")
            else:
                if debug:
                    print(f"[DEBUG] Numeric Grounding retry failed with offending numbers: {retry_offending}")
                final_answer = f"[ERROR: Answer contained ungrounded numbers: {retry_offending}]"

    # Post-answer faithfulness verification if --verify is requested
    if verify and actual_retrieved_chunks and "I couldn't find that in the documents" not in final_answer:
        t_ver = time.time()
        faith_audit = verify_answer_faithfulness(final_answer, actual_retrieved_chunks, model=target_model)
        ver_duration = round((time.time() - t_ver) * 1000, 2)
        if debug:
            print(f"[DEBUG] Faithfulness Audit ({ver_duration}ms): Supported={faith_audit.get('supported')} | Unsupported={faith_audit.get('unsupported_claims')}")

        if not faith_audit.get("supported", True) and faith_audit.get("unsupported_claims"):
            unsupported = faith_audit["unsupported_claims"]
            if debug:
                print(f"[DEBUG] Regenerating answer to remove unsupported claims: {unsupported}")
            regen_prompt = (
                f"Your previous draft contained unsupported claims: {unsupported}. "
                "Regenerate your answer, strictly avoiding these claims and using ONLY the verified context."
            )
            messages.append({"role": "user", "content": regen_prompt})
            regen_resp = execute_chat_with_timeout(
                client=client,
                model=target_model,
                messages=messages,
                options={"temperature": 0.0, "num_ctx": NUM_CTX},
                think=False
            )
            final_answer = regen_resp.message.content or final_answer

    # Strictly match citations in final answer against retrieved chunks
    parsed_cits = extract_citations(final_answer)
    verified_sources = []
    chunk_keys = {f"{c['source_file']}" for c in actual_retrieved_chunks}

    for cit in parsed_cits:
        matching = any(cit.startswith(ck) or ck in cit for ck in chunk_keys)
        if matching:
            verified_sources.append(cit)
        else:
            verified_sources.append(f"UNVERIFIED CITATION: {cit}")

    # Fallback: if no citations parsed in text but chunks were used, not pure invoice, and not refusal, show top chunk
    if not verified_sources and actual_retrieved_chunks and not all_invoices and not is_pure_data and "I couldn't find that in the documents" not in final_answer:
        for c in actual_retrieved_chunks[:2]:
            verified_sources.append(f"{c['source_file']}, p.{c['page']}, §{c['section']}")

    total_latency_ms = round((time.time() - start_time) * 1000, 2)
    invoice_rows = [InvoiceRow(**inv) for inv in all_invoices]
    route_label = f"TOOL_LOOP ({tool_calls_executed} tool calls)" if tool_calls_executed > 0 else "DIRECT_ANSWER"

    # Task 6: Add --debug stage timings: retrieval, forced tools, each ollama.chat call, gate, citation/grounding retries, total. Print slowest stage.
    if debug:
        stages = {
            "retrieval": t_stage_retrieval,
            "forced_tools": t_stage_forced_tools,
            "ollama_chat_calls": t_stage_chat_calls,
            "gate": t_stage_gate,
            "citation_retry": t_stage_cite_retry,
            "grounding_retry": t_stage_ground_retry
        }
        slowest_stage, slowest_time = max(stages.items(), key=lambda x: x[1])
        print(f"\n[DEBUG] Stage Timings:")
        print(f"  retrieval: {t_stage_retrieval:.2f}ms")
        print(f"  forced_tools: {t_stage_forced_tools:.2f}ms")
        print(f"  ollama_chat_calls: {t_stage_chat_calls:.2f}ms ({len(chat_durations)} calls: {[round(d, 1) for d in chat_durations]})")
        print(f"  gate: {t_stage_gate:.2f}ms")
        print(f"  citation_retry: {t_stage_cite_retry:.2f}ms")
        print(f"  grounding_retry: {t_stage_ground_retry:.2f}ms")
        print(f"  total: {total_latency_ms:.2f}ms")
        print(f"  Slowest stage: {slowest_stage} ({slowest_time:.2f}ms)\n")

    return ChatResponse(
        answer=final_answer,
        invoices=invoice_rows,
        sources=list(dict.fromkeys(verified_sources)),
        route=route_label,
        latency_ms=total_latency_ms,
        retrieved_chunks=actual_retrieved_chunks,
        queries_run=queries_run
    )


# ---------------------------------------------------------------------------
# Task 2 & Task 3: Query Planner, Pure Data Templates, & Structured Synthesis
# ---------------------------------------------------------------------------

def get_fallback_plan(question: str, backend: Optional[str] = None) -> QueryPlan:
    """
    Deterministic rule-based fallback planner using regexes and keyword rules.
    Ensures data questions are never routed to policy only.
    """
    q_low = question.lower()
    entities = detect_question_entities(question, backend=backend)

    data_words = [
        "pending", "overdue", "paid", "unpaid", "invoice", "invoices",
        "total value", "how many", "show invoices", "count", "amount", "fee"
    ]
    has_data_keywords = any(dw in q_low for dw in data_words) or bool(entities["invoice_id"]) or bool(entities["vendor"])

    invoice_filter = None
    late_fee_id = None

    if entities["is_late_fee_amount"] and entities["invoice_id"]:
        late_fee_id = entities["invoice_id"]
    elif "late fee" in q_low and entities["invoice_id"] and any(w in q_low for w in ["how much", "owed", "calculate"]):
        late_fee_id = entities["invoice_id"]

    if has_data_keywords or entities["invoice_id"] or entities["vendor"]:
        st = None
        if "unpaid" in q_low or "not paid" in q_low or "outstanding" in q_low:
            st = ["Pending", "Overdue"]
        elif "pending" in q_low:
            st = ["Pending"]
        elif "overdue" in q_low:
            st = ["Overdue"]
        elif "paid" in q_low and "unpaid" not in q_low:
            st = ["Paid"]
        elif "disputed" in q_low:
            st = ["Disputed"]

        min_amt = None
        max_amt = None
        m_lakh = re.search(r'(?:above|over|exceeding|greater than|>)\s*(?:₹|rs\.?)?\s*(\d+(?:\.\d+)?)\s*lakh', q_low)
        if m_lakh:
            min_amt = float(m_lakh.group(1)) * 100000.0
        else:
            m_num = re.search(r'(?:above|over|exceeding|>)\s*(?:₹|rs\.?)?\s*([\d,]+)', q_low)
            if m_num:
                try:
                    min_amt = float(m_num.group(1).replace(",", ""))
                except Exception:
                    pass

        invoice_filter = InvoiceFilter(
            invoice_id=entities["invoice_id"],
            vendor=entities["vendor"],
            status=st,
            min_amount=min_amt,
            max_amount=max_amt
        )

    policy_keywords = [
        "policy", "rule", "period", "term", "terms", "condition", "sla", "escalation",
        "phase", "allowance", "per diem", "withholding", "tds", "gst", "standard",
        "dispute", "procedure", "guideline", "limit", "rate", "salary", "bonus",
        "leave", "travel", "food", "hotel", "cap", "grace", "flight", "disbursement",
        "method", "methods"
    ]
    needs_policy = any(k in q_low for k in policy_keywords) or (not has_data_keywords and not late_fee_id)
    policy_q = question if needs_policy else ""

    return QueryPlan(
        needs_policy=needs_policy,
        policy_query=policy_q,
        invoice=invoice_filter,
        late_fee_invoice_id=late_fee_id
    )


def deterministic_rule_planner(
    question: str,
    backend: Optional[str] = None
) -> Tuple[Optional[QueryPlan], bool]:
    """
    High-confidence deterministic planner:
    - Matches invoice ID regex (e.g. INV-2024-036)
    - Matches vendor names from the DB
    - Classifies status/amount/count queries vs late fee calculations
    - Matches policy and unanswerable keywords
    Returns (QueryPlan, is_confident). If unsure, is_confident is False.
    """
    q_low = question.lower()
    
    # 1. Invoice ID reference parsing:
    canon_inv_id, non_canon_raw, suggested_inv_id = parse_invoice_reference(question)
    
    if non_canon_raw:
        # Non-canonical invoice reference (e.g. INV001, INV 036, invoice number 36, #36):
        # Invoice ID Policy: Use an invoice ID only if written in canonical form (INV-\d{4}-\d{3}).
        # For non-canonical forms, do NOT query the guessed ID. Return not-found for user's text
        # and suggest the closest canonical ID. Never answer from a suggested invoice.
        # Late-fee and payment-method questions with a non-canonical ID follow the same rule.
        return QueryPlan(
            needs_policy=False,
            policy_query="",
            invoice=InvoiceFilter(invoice_id=non_canon_raw),
            late_fee_invoice_id=None
        ), True

    inv_id = canon_inv_id
    
    # 2. Vendor name from DISTINCT vendors
    matched_vendor = None
    try:
        vendors = get_distinct_vendors(backend=backend)
        vendors_sorted = sorted(vendors, key=len, reverse=True)
        for v in vendors_sorted:
            if v.lower() in q_low:
                matched_vendor = v
                break
    except Exception:
        pass
        
    # 3. Late fee calculation intent (must not be asking for policy rate only)
    is_late_fee = False
    if "late fee" in q_low or "interest" in q_low or "penalty" in q_low:
        if any(w in q_low for w in ["calculate", "how much", "owed", "what is the late fee", "what late interest", "what penalty", "charge applies", "penalty applies", "applies to", "apply to"]) and "rate" not in q_low:
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
        "reaches day", "workflow", "paid leave", "payments between", "who approves",
        "how much can a manager approve", "who must approve"
    ]
    has_policy_kw = any(k in q_low for k in policy_keywords)
    
    # 5. Data / invoice keywords
    data_words = [
        "status", "pending", "overdue", "unpaid", "paid", "disputed",
        "invoices", "total amount", "total value", "how many", "count",
        "highest", "show", "list", "exceeding", "above", "below", "due before",
        "running late", "owe", "balance", "vendors are running late"
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
    if inv_id and (has_policy_kw or any(w in q_low for w in ["phase", "escalation", "method", "routing"])):
        if is_late_fee:
            return QueryPlan(needs_policy=True, policy_query=question, late_fee_invoice_id=inv_id, invoice=None), True
        return QueryPlan(needs_policy=True, policy_query=question, invoice=InvoiceFilter(invoice_id=inv_id)), True
        
    # CASE C: Pure invoice lookup by ID
    if inv_id and not has_policy_kw:
        return QueryPlan(needs_policy=False, invoice=InvoiceFilter(invoice_id=inv_id)), True
        
    # CASE D: Vendor invoice query (including how much do we owe)
    if matched_vendor and not has_policy_kw and not has_unans:
        st = None
        if "unpaid" in q_low or "not paid" in q_low or "outstanding" in q_low or "owe" in q_low or "balance" in q_low:
            st = ["Pending", "Overdue"]
        elif "pending" in q_low:
            st = ["Pending"]
        elif "overdue" in q_low or "late" in q_low:
            st = ["Overdue"]
        elif "paid" in q_low:
            st = ["Paid"]
        elif "disputed" in q_low:
            st = ["Disputed"]
        return QueryPlan(needs_policy=False, invoice=InvoiceFilter(vendor=matched_vendor, status=st)), True

    # CASE E1: Vendors running late / overdue vendors
    if any(w in q_low for w in ["running late", "overdue vendors", "vendors are running late", "which vendors are late", "which vendors are running late"]):
        return QueryPlan(needs_policy=False, invoice=InvoiceFilter(status=["Overdue"])), True
        
    # CASE E2: Pure aggregate invoice query (status / counts / amounts)
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


def validate_and_guard_llm_plan(plan: QueryPlan, question: str, backend: Optional[str] = None) -> QueryPlan:
    r"""
    Validates any LLM plan against the question text:
    - Keep an invoice filter only if the question contains:
      * a canonical invoice ID (regex INV-\d{4}-\d{3}, case-insensitive)
      * a vendor name from the DB
      * invoice-data words (invoice, invoices, overdue, unpaid, pending, amount owed)
    - 'paid leave' must NOT create a status filter or invoice data filter.
    - Drops unsupported filters and logs warnings.
    """
    q_low = question.lower()

    # 1. Check canonical or non-canonical invoice ID
    canonical_match = re.search(r"\bINV-\d{4}-\d{3}\b", question, re.I)
    non_canonical_match = re.search(r"\b(?:INV\s*\d+|invoice\s*(?:number|no\.?|#)\s*\d+|#\d+)\b", question, re.I)

    # 2. Check vendor name from DB (matching full name or vendor tokens > 3 chars like 'deloitte')
    has_vendor = False
    try:
        vendors = get_distinct_vendors(backend=backend)
        for v in vendors:
            v_low = v.lower()
            if (len(v_low) > 2 and v_low in q_low) or any(len(tok) > 3 and tok in q_low for tok in v_low.split()):
                has_vendor = True
                break
    except Exception as e:
        logger.warning(f"Error fetching vendors for plan validation: {e}")

    # 3. Check invoice-data words
    invoice_data_words = ["invoice", "invoices", "overdue", "unpaid", "pending", "amount owed", "owe", "owed", "late", "running late", "vendor", "vendors", "balance"]
    has_invoice_data_words = any(w in q_low for w in invoice_data_words)

    # Special guard: 'paid leave' is an HR / leave policy term and must not count as invoice data intent
    is_paid_leave = "paid leave" in q_low

    # Does question have any valid invoice grounding?
    has_invoice_grounding = (bool(canonical_match) or bool(non_canonical_match) or has_vendor or has_invoice_data_words) and not (
        is_paid_leave and not (canonical_match or has_vendor or any(w in q_low for w in ["invoice", "invoices"]))
    )

    if plan.invoice is not None:
        if not has_invoice_grounding:
            logger.warning(f"[LLM-PLANNER GUARD] Dropping unsupported invoice filter (no canonical ID, vendor, or invoice words in question): '{question}'")
            plan.invoice = None
        else:
            # Drop unsupported status filter created by "paid leave"
            if is_paid_leave and plan.invoice.status:
                orig_st = list(plan.invoice.status)
                plan.invoice.status = [s for s in plan.invoice.status if s.lower() != "paid"]
                if not plan.invoice.status:
                    plan.invoice.status = None
                if orig_st != plan.invoice.status:
                    logger.warning(f"[LLM-PLANNER GUARD] Dropped unsupported 'Paid' status filter derived from 'paid leave': '{question}'")

            # Guard policy / workflow questions from hallucinated invoice filters
            # e.g., "overdue escalation workflow", "what happens when an invoice reaches day 15 overdue", "dispute resolution procedure", "retained under compliance"
            policy_workflow_kws = ["escalation workflow", "escalation procedure", "dispute resolution", "what happens when an invoice reaches", "retained under compliance", "retention policy"]
            if any(pw in q_low for pw in policy_workflow_kws) and not any(dq in q_low for dq in ["which invoices", "show invoices", "list invoices", "how many"]):
                if plan.invoice.status or plan.invoice.invoice_id is None:
                    logger.warning(f"[LLM-PLANNER GUARD] Dropping unsupported invoice filter for policy/workflow question: '{question}'")
                    plan.invoice = None

            # Validate vendor filter against question text
            if plan.invoice and plan.invoice.vendor:
                v_target = plan.invoice.vendor.lower()
                if v_target not in q_low and not any(len(tok) > 3 and tok in q_low for tok in v_target.split()):
                    logger.warning(f"[LLM-PLANNER GUARD] Dropping hallucinated vendor filter '{plan.invoice.vendor}' not in question: '{question}'")
                    plan.invoice.vendor = None

            # Validate invoice_id filter against question text
            if plan.invoice and plan.invoice.invoice_id:
                inv_raw = plan.invoice.invoice_id.lower().replace("-", "").replace(" ", "")
                q_clean = q_low.replace("-", "").replace(" ", "")
                if inv_raw not in q_clean and not any(part in q_clean for part in inv_raw.split() if len(part) > 1):
                    logger.warning(f"[LLM-PLANNER GUARD] Dropping hallucinated invoice_id '{plan.invoice.invoice_id}' not in question: '{question}'")
                    plan.invoice.invoice_id = None

            # Validate amounts against question text
            if plan.invoice and plan.invoice.min_amount is not None and not re.search(r"\d+", q_low):
                logger.warning(f"[LLM-PLANNER GUARD] Dropping unsupported min_amount '{plan.invoice.min_amount}' not in question: '{question}'")
                plan.invoice.min_amount = None
            if plan.invoice and plan.invoice.max_amount is not None and not re.search(r"\d+", q_low):
                logger.warning(f"[LLM-PLANNER GUARD] Dropping unsupported max_amount '{plan.invoice.max_amount}' not in question: '{question}'")
                plan.invoice.max_amount = None

            # If all fields in plan.invoice are None/empty, drop plan.invoice completely
            if plan.invoice:
                has_active_fields = any([
                    plan.invoice.invoice_id,
                    plan.invoice.vendor,
                    plan.invoice.status,
                    plan.invoice.min_amount is not None,
                    plan.invoice.max_amount is not None,
                    plan.invoice.due_before
                ])
                if not has_active_fields:
                    logger.warning(f"[LLM-PLANNER GUARD] Dropping empty invoice filter: '{question}'")
                    plan.invoice = None

    return plan


def generate_query_plan(
    question: str,
    client: Any,
    model: str,
    backend: Optional[str] = None,
    planner: str = "auto",
    debug: bool = False
) -> Tuple[QueryPlan, str, float]:
    """
    Invokes deterministic rules first when planner in ('rules', 'auto').
    If confident, returns immediately in < 1ms.
    If unsure (or planner == 'llm'), invokes Ollama with QueryPlan schema.
    Validates JSON with Pydantic; falls back to deterministic rules if invalid or failed.
    """
    t0 = time.time()
    if planner in ("rules", "auto"):
        r_plan, confident = deterministic_rule_planner(question, backend=backend)
        if confident:
            plan_dur = round((time.time() - t0) * 1000, 2)
            return r_plan, "rules", plan_dur
        if planner == "rules":
            plan_dur = round((time.time() - t0) * 1000, 2)
            fallback = get_fallback_plan(question, backend=backend)
            return fallback, "rules_fallback", plan_dur

    try:
        resp = execute_chat_with_timeout(
            client=client,
            model=model,
            messages=[
                {"role": "system", "content": QUERY_PLANNER_PROMPT.format(question=question)},
                {"role": "user", "content": f"Generate execution plan for question: {question}"}
            ],
            format=QueryPlan.model_json_schema(),
            options={"temperature": 0.0, "num_predict": 150, "num_ctx": NUM_CTX},
            think=False,
            keep_alive="30m"
        )
        plan_dur = round((time.time() - t0) * 1000, 2)
        content = resp.message.content or ""
        parsed = json.loads(content)
        plan = QueryPlan.model_validate(parsed)

        q_low = question.lower()
        # Normalize status mapping: unpaid -> ["Pending", "Overdue"]
        if plan.invoice:
            if plan.invoice.status:
                clean_st = []
                for s in plan.invoice.status:
                    s_str = str(s).strip().lower()
                    if s_str in ("unpaid", "not paid", "outstanding"):
                        clean_st.extend(["Pending", "Overdue"])
                    elif s_str == "pending":
                        clean_st.append("Pending")
                    elif s_str == "overdue":
                        clean_st.append("Overdue")
                    elif s_str == "paid":
                        clean_st.append("Paid")
                    elif s_str == "disputed":
                        clean_st.append("Disputed")
                    elif s_str in ("partially paid", "partially_paid", "partial"):
                        clean_st.append("Partially Paid")
                    else:
                        clean_st.append(str(s))
                plan.invoice.status = list(dict.fromkeys(clean_st))
            else:
                # If question mentions unpaid/pending/overdue/paid but status was omitted
                if "unpaid" in q_low or "not paid" in q_low or "outstanding" in q_low:
                    plan.invoice.status = ["Pending", "Overdue"]
                elif "pending" in q_low:
                    plan.invoice.status = ["Pending"]
                elif "overdue" in q_low:
                    plan.invoice.status = ["Overdue"]
                elif "paid" in q_low and "unpaid" not in q_low:
                    plan.invoice.status = ["Paid"]

            # Guard against hallucinated bounds
            if plan.invoice.max_amount and not any(w in q_low for w in ["below", "under", "less than", "maximum", "max", "<"]):
                plan.invoice.max_amount = None
            if plan.invoice.min_amount and not any(w in q_low for w in ["above", "over", "greater than", "exceeding", "more than", "min", "minimum", ">"]):
                if not re.search(r'\d+', q_low):
                    plan.invoice.min_amount = None
            if plan.invoice.due_before:
                date_words = ["before", "by", "until", "prior", "due date", "202", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
                if not any(dw in q_low for dw in date_words):
                    plan.invoice.due_before = None

        # Guard against routing non-invoice questions (e.g. bank account number, salary, address) to invoice DB
        non_invoice_topics = ["bank account", "account number", "routing number", "salary", "personal address", "dress code", "tax rate"]
        if any(nit in q_low for nit in non_invoice_topics):
            plan.invoice = None
            plan.needs_policy = True
            if not plan.policy_query:
                plan.policy_query = question

        # Apply LLM plan guard
        plan = validate_and_guard_llm_plan(plan, question, backend=backend)

        # Policy intent check
        policy_intent_words = [
            "payment method", "payment methods", "escalation", "phase",
            "per diem", "allowance", "withholding", "tds rate", "gst rate",
            "policy", "terms", "sla", "dispute", "rule", "guideline", "condition",
            "disbursement", "disbursements", "three-way", "matching", "retention", "retain", "retained"
        ]
        has_policy_intent = any(pw in q_low for pw in policy_intent_words)
        if has_policy_intent and not plan.needs_policy:
            plan.needs_policy = True
            if not plan.policy_query:
                plan.policy_query = question

        # Fallback safeguard: data intent must never be routed to policy only!
        # Do not augment policy-only questions with invoice data filters
        is_querying_specific_invoices = any(dq in q_low for dw in ["which invoices", "show invoices", "list invoices", "how many invoices"] if dw in q_low)
        has_data_intent = any(dw in q_low for dw in [
            "pending", "overdue", "paid", "unpaid", "total value",
            "how many invoices", "show invoices", "invoices above", "invoices below", "invoice"
        ]) and not any(nit in q_low for nit in non_invoice_topics) and ("paid leave" not in q_low) and (not has_policy_intent or is_querying_specific_invoices)

        if has_data_intent and plan.invoice is None and plan.late_fee_invoice_id is None:
            fallback = get_fallback_plan(question, backend=backend)
            plan.invoice = fallback.invoice
            if fallback.late_fee_invoice_id:
                plan.late_fee_invoice_id = fallback.late_fee_invoice_id
            plan = validate_and_guard_llm_plan(plan, question, backend=backend)
            return plan, "fallback_augmented", plan_dur

        # Final guard pass
        plan = validate_and_guard_llm_plan(plan, question, backend=backend)

        # Fallback safeguard: if plan has NO actions at all (needs_policy is False, invoice is None, late_fee is None),
        # an empty plan is invalid — fall back to rule-based planner.
        if not plan.needs_policy and plan.invoice is None and plan.late_fee_invoice_id is None:
            fallback = get_fallback_plan(question, backend=backend)
            return fallback, "fallback", plan_dur

        return plan, "planner", plan_dur
    except Exception as e:
        plan_dur = round((time.time() - t0) * 1000, 2)
        fallback = get_fallback_plan(question, backend=backend)
        return fallback, "fallback", plan_dur


def render_pure_data_answer(
    tool_data_results: List[Any],
    latest_fee_data: Optional[Dict[str, Any]],
    question: str
) -> str:
    """
    Renders pure data answers deterministically via Python templates.
    Handles counts, total with Indian grouping and crore/lakh notation, status breakdown, and single rows.
    """
    if latest_fee_data and latest_fee_data.get("found"):
        fee_text = format_late_fee_template(latest_fee_data)
        inv_results = [t for t in tool_data_results if isinstance(t, dict) and "invoices" in t]
        if not inv_results:
            return fee_text
        inv = inv_results[0]["invoices"][0] if inv_results[0].get("invoices") else None
        if inv:
            amt_str = format_inr(inv["amount_inr"])
            return f"Invoice {inv['invoice_id']} for vendor {inv['vendor_name']} has a status of {inv['status']} with an amount of {amt_str}.\n\n{fee_text}"
        return fee_text

    inv_results = [t for t in tool_data_results if isinstance(t, dict) and ("invoices" in t or "searched_id" in t or "found" in t)]
    if not inv_results:
        return "I couldn't find that in the documents."

    inv_res = inv_results[0]
    if not inv_res.get("found", True):
        return inv_res.get("error") or "Invoice was not found in the records."

    invoices = inv_res.get("invoices", [])
    count = inv_res.get("count", len(invoices))
    total = inv_res.get("total_amount", sum(float(i.get("amount_inr", 0)) for i in invoices))

    # Single invoice lookup
    if count == 1 and invoices:
        inv = invoices[0]
        amt_str = format_inr(inv["amount_inr"])
        due_str = f" Due date is {inv['due_date']}." if inv.get("due_date") else ""
        paid_str = f" Paid on {inv['payment_date']}." if inv.get("payment_date") else ""
        return (
            f"Invoice {inv['invoice_id']} for vendor {inv['vendor_name']} has a status of {inv['status']} "
            f"with an amount of {amt_str}.{due_str}{paid_str}"
        )

    # Multi-invoice summary
    status_breakdown = inv_res.get("status_breakdown", {})
    if not status_breakdown and invoices:
        sb = {}
        for inv in invoices:
            st = inv.get("status", "Unknown")
            sb[st] = sb.get(st, 0) + 1
        status_breakdown = sb

    sb_parts = [f"{st}: {c}" for st, c in sorted(status_breakdown.items())]
    sb_str = ", ".join(sb_parts) if sb_parts else "None"

    total_inr = format_inr(total)
    crore_val = total / 1e7
    lakh_val = total / 1e5
    crore_str = f"{crore_val:.2f} crore"
    lakh_str = f"{lakh_val:.2f} lakh"

    unit_str = f" ({crore_str})" if total >= 1e7 else (f" ({lakh_str})" if total >= 1e5 else "")
    lines = [
        f"There are {count} matching invoices with a total value of {total_inr}{unit_str}.",
        f"Status breakdown: {sb_str}."
    ]
    return "\n".join(lines)


def verify_chunk_citations(
    cited_ids: List[int],
    answer: str,
    chunks: List[Dict[str, Any]]
) -> List[int]:
    """
    Checks each cited chunk id (1-based index) by lexical overlap with the answer,
    dropping unsupported ones.
    """
    valid_ids: List[int] = []
    ans_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', answer.lower()))
    stop_words = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has", "have", "had", "not", "but"}
    ans_words = {w for w in ans_words if w not in stop_words}

    for cid in cited_ids:
        idx = cid - 1
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            chunk_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', chunk.get("content", "").lower()))
            overlap = ans_words.intersection(chunk_words)
            if len(overlap) >= 3:
                valid_ids.append(cid)
    return valid_ids


def format_tool_payload_for_prompt(tool_data: List[Any], plan: QueryPlan) -> Tuple[List[Any], str]:
    """
    Guards tool payload:
    - Policy-only plans must never inject invoice rows.
    - Plans with invoice rows inject at most 10 rows (with summary string for excess).
    """
    is_policy_only = plan.needs_policy and not plan.invoice and not plan.late_fee_invoice_id
    if is_policy_only:
        return [], "None"

    sanitized = []
    for t in tool_data:
        if isinstance(t, dict) and "invoices" in t:
            invs = t.get("invoices", [])
            total_count = t.get("count", len(invs))
            if len(invs) > 10:
                t_sub = dict(t)
                t_sub["invoices"] = invs[:10]
                t_sub["summary"] = f"Displaying top 10 of {total_count} matching invoice rows. {total_count - 10} additional rows omitted."
                sanitized.append(t_sub.get("model_payload", t_sub))
            else:
                sanitized.append(t.get("model_payload", t))
        else:
            sanitized.append(t.get("model_payload", t) if isinstance(t, dict) else t)

    return sanitized, json.dumps(sanitized, default=str) if sanitized else "None"


def chat(
    question: str,
    model: Optional[str] = None,
    router: str = "llm",
    backend: Optional[str] = None,
    retrieval_strategy: str = "hybrid",
    retrieval_mode: str = "always",
    keyword_bypass: bool = False,
    answer_gate: bool = True,
    gate_mode: str = "separate",
    gate_model: Optional[str] = None,
    use_reranker: bool = False,
    verify: bool = False,
    legacy_tools: bool = False,
    llm_data_answers: bool = False,
    embed_model: Optional[str] = None,
    planner: str = "auto",
    debug: bool = False
) -> ChatResponse:
    """
    Main Assistant chat handler.
    Default path: Query Planner -> Plan Execution -> Template / Structured Synthesis.
    Legacy tool loop preserved behind --legacy-tools.
    """
    if legacy_tools:
        return _chat_legacy_tools(
            question=question,
            model=model,
            router=router,
            backend=backend,
            retrieval_strategy=retrieval_strategy,
            retrieval_mode=retrieval_mode,
            keyword_bypass=keyword_bypass,
            answer_gate=answer_gate,
            gate_model=gate_model,
            use_reranker=use_reranker,
            verify=verify,
            debug=debug
        )

    start_time = time.time()
    target_model = model or CHAT_MODEL
    client = get_ollama_client()

    t_stage_planner = 0.0
    t_stage_retrieval = 0.0
    t_stage_forced_tools = 0.0
    t_stage_chat_calls = 0.0
    t_stage_gate = 0.0
    t_stage_ground_retry = 0.0
    chat_durations: List[float] = []
    ollama_call_count = 0

    all_invoices: List[Dict[str, Any]] = []
    executed_tools: List[Dict[str, Any]] = []
    actual_retrieved_chunks: List[Dict[str, Any]] = []
    queries_run: List[str] = []
    latest_fee_data: Optional[Dict[str, Any]] = None
    tool_data_results: List[Any] = []

    # Step 1: Query Planner
    plan, plan_source, plan_dur = generate_query_plan(question, client, target_model, backend=backend, planner=planner, debug=debug)
    t_stage_planner = plan_dur
    if plan_source == "planner":
        ollama_call_count += 1
        chat_durations.append(plan_dur)

    if debug:
        print(f"[DEBUG] Plan ({plan_source}): {plan.model_dump_json()}")

    # Step 2: Execute Plan
    if plan.late_fee_invoice_id:
        t0_f = time.time()
        fee_res = calculate_late_fee(invoice_id=plan.late_fee_invoice_id, backend=backend)
        t_stage_forced_tools += (time.time() - t0_f) * 1000
        latest_fee_data = fee_res
        tool_data_results.append(fee_res)
        executed_tools.append({"name": "calculate_late_fee", "args": {"invoice_id": plan.late_fee_invoice_id}, "forced": False})
        if debug:
            print(f"[DEBUG] Tool Call: calculate_late_fee({json.dumps({'invoice_id': plan.late_fee_invoice_id})})")

    if plan.invoice:
        inv_args: Dict[str, Any] = {}
        if plan.invoice.invoice_id:
            inv_args["invoice_id"] = plan.invoice.invoice_id
        if plan.invoice.vendor:
            inv_args["vendor"] = plan.invoice.vendor
        if plan.invoice.status:
            inv_args["status"] = plan.invoice.status
        if plan.invoice.min_amount is not None:
            inv_args["min_amount"] = plan.invoice.min_amount
        if plan.invoice.max_amount is not None:
            inv_args["max_amount"] = plan.invoice.max_amount
        if plan.invoice.due_before:
            inv_args["due_before"] = plan.invoice.due_before

        t0_i = time.time()
        inv_res = query_invoices(backend=backend, **inv_args)
        t_stage_forced_tools += (time.time() - t0_i) * 1000
        tool_data_results.append(inv_res)
        if inv_res.get("invoices"):
            all_invoices.extend(inv_res["invoices"])
        executed_tools.append({"name": "query_invoices", "args": inv_args, "forced": False})
        if debug:
            print(f"[DEBUG] Tool Call: query_invoices({json.dumps(inv_args)})")

    if plan.needs_policy:
        p_query = plan.policy_query.strip() or question
        queries_run.append(p_query)
        executed_tools.append({"name": "search_policies", "args": {"query": p_query}, "forced": False})
        if debug:
            print(f"[DEBUG] Tool Call: search_policies({json.dumps({'query': p_query})})")
        t0_ret = time.time()
        chunks, emb_ms, ret_ms = search_chunks_with_timing(
            query=p_query,
            limit=5,
            retrieval_mode=retrieval_strategy,
            use_reranker=use_reranker,
            backend=backend,
            embed_model=embed_model
        )
        t_stage_retrieval += (time.time() - t0_ret) * 1000
        actual_retrieved_chunks.extend(chunks)

    # Step 3: Answer Generation
    data_ok = any(
        (isinstance(t, dict) and (t.get("found") is True or t.get("count", 0) > 0 or len(t.get("invoices", [])) > 0))
        for t in tool_data_results
    )
    not_found_ok = any(
        (isinstance(t, dict) and t.get("found") is False and ("searched_id" in t or "suggested_ids" in t))
        for t in tool_data_results
    )
    is_pure_data = (data_ok or not_found_ok) and not plan.needs_policy

    final_answer = ""
    gate_supported = True
    gate_quote = ""
    gate_chunk_idx: Optional[int] = None
    gate_model_target = gate_model or GATE_MODEL
    valid_cids: List[int] = []

    if is_pure_data and not llm_data_answers:
        final_answer = render_pure_data_answer(tool_data_results, latest_fee_data, question)
        if debug:
            print("[DEBUG] Rendered answer via deterministic Python template.")
    else:
        # Policy or mixed question, or --llm-data-answers
        if gate_mode == "separate":
            if answer_gate and actual_retrieved_chunks and not is_pure_data:
                t0_g = time.time()
                gate_supported, gate_quote, gate_chunk_idx = check_answerability_gate(
                    question=question,
                    chunks=actual_retrieved_chunks,
                    tool_results=tool_data_results,
                    gate_model=gate_model_target,
                    debug=debug
                )
                gate_dur = round((time.time() - t0_g) * 1000, 2)
                t_stage_gate = gate_dur
                ollama_call_count += 1
                chat_durations.append(gate_dur)

            # Refusal rule: refuse with "I couldn't find that in the documents."
            # only if NOT data_ok, NOT not_found_ok, AND NOT gate_supported.
            if (not data_ok) and (not not_found_ok) and (not gate_supported):
                final_answer = "I couldn't find that in the documents."
                if debug:
                    print(f"[DEBUG] Refusal triggered: gate unsupported.")
            else:
                context_blocks = [
                    f"[{i+1}] [{c['source_file']}, p.{c['page']}, §{c['section']}]:\n{c['content']}"
                    for i, c in enumerate(actual_retrieved_chunks)
                ]
                ctx_str = "\n\n".join(context_blocks) if context_blocks else "None"
                is_policy_only = plan.needs_policy and not plan.invoice and not plan.late_fee_invoice_id
                _, tool_str = format_tool_payload_for_prompt(tool_data_results, plan)
                numeric_facts, matching_quote = build_verified_numeric_facts(
                    question=question,
                    chunks=actual_retrieved_chunks,
                    tool_results=[] if is_policy_only else tool_data_results
                )
                num_facts_block = f"\n{numeric_facts}\n" if numeric_facts else ""
                s_prompt = STRUCTURED_ANSWER_PROMPT.format(
                    context=ctx_str,
                    tool_results=tool_str,
                    numeric_facts=num_facts_block,
                    question=question
                )
                t_syn = time.time()
                resp = execute_chat_with_timeout(
                    client=client,
                    model=target_model,
                    messages=[{"role": "user", "content": s_prompt}],
                    format=StructuredAnswer.model_json_schema(),
                    options={"temperature": 0.0, "num_ctx": NUM_CTX},
                    think=False,
                    keep_alive="30m"
                )
                syn_dur = round((time.time() - t_syn) * 1000, 2)
                t_stage_chat_calls += syn_dur
                ollama_call_count += 1
                chat_durations.append(syn_dur)

                try:
                    s_ans = StructuredAnswer.model_validate_json(resp.message.content)
                    final_answer = s_ans.answer
                    q_ver, q_idx = verify_quote_against_chunks(s_ans.supporting_quote, actual_retrieved_chunks)
                    if q_ver and q_idx is not None and gate_chunk_idx is None:
                        gate_chunk_idx = q_idx
                    valid_cids = verify_chunk_citations(s_ans.cited_chunk_ids, final_answer, actual_retrieved_chunks)
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Structured answer parsing failed ({e}), using raw text.")
                    final_answer = resp.message.content or ""
                    valid_cids = []

        elif gate_mode == "merged":
            context_blocks = [
                f"[{i+1}] [{c['source_file']}, p.{c['page']}, §{c['section']}]:\n{c['content']}"
                for i, c in enumerate(actual_retrieved_chunks)
            ]
            ctx_str = "\n\n".join(context_blocks) if context_blocks else "None"
            is_policy_only = plan.needs_policy and not plan.invoice and not plan.late_fee_invoice_id
            _, tool_str = format_tool_payload_for_prompt(tool_data_results, plan)
            numeric_facts, matching_quote = build_verified_numeric_facts(
                question=question,
                chunks=actual_retrieved_chunks,
                tool_results=[] if is_policy_only else tool_data_results
            )
            num_facts_block = f"\n{numeric_facts}\n" if numeric_facts else ""
            s_prompt = STRUCTURED_ANSWER_PROMPT.format(
                context=ctx_str,
                tool_results=tool_str,
                numeric_facts=num_facts_block,
                question=question
            )
            t_syn = time.time()
            resp = execute_chat_with_timeout(
                client=client,
                model=target_model,
                messages=[{"role": "user", "content": s_prompt}],
                format=StructuredAnswer.model_json_schema(),
                options={"temperature": 0.0, "num_ctx": NUM_CTX},
                think=False,
                keep_alive="30m"
            )
            syn_dur = round((time.time() - t_syn) * 1000, 2)
            t_stage_chat_calls += syn_dur
            ollama_call_count += 1
            chat_durations.append(syn_dur)

            try:
                s_ans = StructuredAnswer.model_validate_json(resp.message.content)
                q_ver, q_idx = verify_quote_against_chunks(s_ans.supporting_quote, actual_retrieved_chunks)
                ans_lower = s_ans.answer.lower()
                q_lower = question.lower()
                has_cited_chunk = bool(actual_retrieved_chunks) and (
                    (q_ver and q_idx is not None) or
                    bool(s_ans.cited_chunk_ids and 0 <= s_ans.cited_chunk_ids[0] - 1 < len(actual_retrieved_chunks))
                )
                rule_elements = ["cap", "grace period", "grace", "exception", "limit", "ceiling", "maximum"]
                asks_rule_element = any(re.search(rf"\b{re.escape(re_term)}\b", q_lower) for re_term in rule_elements)
                gap_phrases = ["not specif", "does not mention", "doesn't mention", "no mention", "not state", "not address", "no cap", "no grace", "does not prescribe"]
                says_not_specified = any(phrase in ans_lower for phrase in gap_phrases)

                is_gap_answer = False
                if has_cited_chunk and asks_rule_element and says_not_specified:
                    c_idx = q_idx if (q_ver and q_idx is not None) else (s_ans.cited_chunk_ids[0] - 1)
                    candidate_chunk = actual_retrieved_chunks[c_idx]
                    c_content_lower = candidate_chunk.get("content", "").lower()
                    q_topic_words = [
                        w for w in re.findall(r"\b[a-zA-Z]{4,}\b", q_lower)
                        if w not in {
                            "what", "when", "where", "which", "there", "does", "with", "from", "that", "this",
                            "about", "before", "after", "starts", "accruing", "apply", "have", "been", "rule",
                            "policy", "documents", "specified", "mention"
                        } and not any(r in w for r in ["cap", "grace", "limit", "exception", "ceiling", "maximum"])
                    ]
                    topic_in_chunk = any(tw in c_content_lower for tw in q_topic_words) if q_topic_words else True
                    if topic_in_chunk:
                        is_gap_answer = True

                if not s_ans.answerable or (not q_ver and not data_ok):
                    if not data_ok and not not_found_ok and not is_gap_answer:
                        final_answer = "I couldn't find that in the documents."
                    else:
                        final_answer = s_ans.answer
                else:
                    final_answer = s_ans.answer
                if q_ver and q_idx is not None:
                    gate_chunk_idx = q_idx
                elif s_ans.cited_chunk_ids and 0 <= s_ans.cited_chunk_ids[0] - 1 < len(actual_retrieved_chunks):
                    gate_chunk_idx = s_ans.cited_chunk_ids[0] - 1
                valid_cids = verify_chunk_citations(s_ans.cited_chunk_ids, final_answer, actual_retrieved_chunks)
            except Exception as e:
                final_answer = resp.message.content or ""
                valid_cids = []

        # Citations handling
        if "I couldn't find that in the documents" not in final_answer:
            if is_pure_data or (data_ok and not gate_supported and not plan.needs_policy):
                final_answer = re.sub(r'\[[a-zA-Z0-9_\-\.]+\.pdf[^\]]*\]', '', final_answer).strip()
            else:
                cits_to_add = []
                if gate_chunk_idx is not None and gate_chunk_idx < len(actual_retrieved_chunks):
                    vc = actual_retrieved_chunks[gate_chunk_idx]
                    cit_str = f"[{vc['source_file']}, p.{vc['page']}, §{vc['section']}]"
                    if cit_str not in final_answer and vc['source_file'] not in final_answer:
                        cits_to_add.append(cit_str)
                for cid in valid_cids:
                    idx = cid - 1
                    if 0 <= idx < len(actual_retrieved_chunks):
                        vc = actual_retrieved_chunks[idx]
                        cit_str = f"[{vc['source_file']}, p.{vc['page']}, §{vc['section']}]"
                        if cit_str not in final_answer and cit_str not in cits_to_add and vc['source_file'] not in final_answer:
                            cits_to_add.append(cit_str)
                if cits_to_add:
                    final_answer = f"{final_answer}\n\n" + " ".join(cits_to_add)

        # Numeric Grounding Check
        if "I couldn't find that in the documents" not in final_answer:
            all_tool_outputs = [json.dumps(t, default=str) for t in tool_data_results]
            num_ok, offending = check_numeric_grounding(
                answer=final_answer,
                tool_results=all_tool_outputs,
                chunks=actual_retrieved_chunks,
                question=question
            )
            if not num_ok:
                if debug:
                    print(f"[DEBUG] Numeric Grounding Check failed. Offending: {offending}. Retrying once...")
                regen_prompt = (
                    f"Your previous answer contained ungrounded numbers or ₹ amounts not supported by the context: {offending}. "
                    "Rewrite your answer, strictly using ONLY the exact numbers present in the retrieved tool and document context."
                )
                t_num = time.time()
                retry_resp = execute_chat_with_timeout(
                    client=client,
                    model=target_model,
                    messages=[
                        {"role": "user", "content": s_prompt if 's_prompt' in locals() else question},
                        {"role": "assistant", "content": final_answer},
                        {"role": "user", "content": regen_prompt}
                    ],
                    options={"temperature": 0.0, "num_ctx": NUM_CTX},
                    think=False,
                    keep_alive="30m"
                )
                g_dur = round((time.time() - t_num) * 1000, 2)
                t_stage_ground_retry += g_dur
                ollama_call_count += 1
                chat_durations.append(g_dur)
                retry_ans = retry_resp.message.content or ""
                retry_ok, retry_off = check_numeric_grounding(
                    answer=retry_ans,
                    tool_results=all_tool_outputs,
                    chunks=actual_retrieved_chunks,
                    question=question
                )
                if retry_ok:
                    final_answer = retry_ans

        # Deterministic Missing Condition Check (Part B)
        if "I couldn't find that in the documents" not in final_answer and actual_retrieved_chunks:
            from app.utils.deterministic_guard import find_cited_chunk, check_missing_condition
            cited_chunk = find_cited_chunk(final_answer, actual_retrieved_chunks)
            if not cited_chunk and gate_chunk_idx is not None and gate_chunk_idx < len(actual_retrieved_chunks):
                cited_chunk = actual_retrieved_chunks[gate_chunk_idx]
            if not cited_chunk and actual_retrieved_chunks:
                cited_chunk = actual_retrieved_chunks[0]
            cond_flag = check_missing_condition(final_answer, cited_chunk)
            if cond_flag:
                if debug:
                    print(f"[DEBUG] Missing Condition Check flagged: {cond_flag}. Deterministically appending condition...")
                m_cond = re.search(r"conditioned by ['\"](.*?)['\"], but answer", cond_flag, re.DOTALL)
                cond_quote = re.sub(r"\s+", " ", m_cond.group(1)).strip() if m_cond else ""
                if cond_quote:
                    final_answer = append_condition_to_answer(final_answer, cond_quote, cited_chunk)

        # Deterministic Data Claim Guard (Part B.2)
        if tool_data_results and "I couldn't find that in the documents" not in final_answer:
            from app.utils.deterministic_guard import verify_and_guard_data_claims
            final_answer = verify_and_guard_data_claims(final_answer, tool_data_results)

        # Log-only deterministic guard checks in debug mode (Part C)
        if debug and actual_retrieved_chunks:
            from app.utils.deterministic_guard import check_deterministic_faithfulness
            g_audit = check_deterministic_faithfulness(final_answer, actual_retrieved_chunks, tool_results=tool_data_results, question=question)
            non_num_flags = [f for f in g_audit.get("flags", []) if not f.startswith("UNGROUNDED_NUMBERS")]
            if non_num_flags:
                print(f"[DEBUG WARNING] Deterministic guard flags (log-only): {non_num_flags}")

    # Verified sources
    parsed_cits = extract_citations(final_answer)
    verified_sources = []
    chunk_keys = {f"{c['source_file']}" for c in actual_retrieved_chunks}

    for cit in parsed_cits:
        matching = any(cit.startswith(ck) or ck in cit for ck in chunk_keys)
        if matching:
            verified_sources.append(cit)
        else:
            verified_sources.append(f"UNVERIFIED CITATION: {cit}")

    if not verified_sources and actual_retrieved_chunks and plan.needs_policy and "I couldn't find that in the documents" not in final_answer:
        for c in actual_retrieved_chunks[:2]:
            verified_sources.append(f"{c['source_file']}, p.{c['page']}, §{c['section']}")

    total_latency_ms = round((time.time() - start_time) * 1000, 2)
    invoice_rows = [InvoiceRow(**inv) for inv in all_invoices]
    route_label = f"PLANNER ({plan_source})"

    return ChatResponse(
        answer=final_answer,
        invoices=invoice_rows,
        sources=list(dict.fromkeys(verified_sources)),
        route=route_label,
        latency_ms=total_latency_ms,
        retrieved_chunks=actual_retrieved_chunks,
        queries_run=queries_run,
        plan=plan.model_dump(),
        plan_source=plan_source,
        tool_results=tool_data_results,
        ollama_calls=ollama_call_count,
        stage_timings={
            "planner": t_stage_planner,
            "retrieval": t_stage_retrieval,
            "forced_tools": t_stage_forced_tools,
            "gate": t_stage_gate,
            "synthesis": t_stage_chat_calls,
            "grounding_retry": t_stage_ground_retry,
            "total": total_latency_ms
        },
        prompt_tokens=get_last_chat_prompt_tokens()
    )

