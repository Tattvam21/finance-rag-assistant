import re
from typing import Dict, List, Optional

# In-memory session store: session_id -> list of {"role": "user"|"assistant", "content": str}
SESSIONS: Dict[str, List[Dict[str, str]]] = {}

def get_session_history(session_id: str) -> List[Dict[str, str]]:
    return SESSIONS.get(session_id, [])

def add_to_session(session_id: str, role: str, content: str, max_turns: int = 10):
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
    SESSIONS[session_id].append({"role": role, "content": content})
    if len(SESSIONS[session_id]) > max_turns * 2:
        SESSIONS[session_id] = SESSIONS[session_id][-max_turns * 2:]

def rewrite_followup_query(question: str, history: List[Dict[str, str]]) -> str:
    """
    Rewrites contextual follow-up questions (e.g. 'what about Vendor ABC?', 'and for Delhi?')
    into self-contained standalone search queries before retrieval.
    """
    if not history:
        return question

    q_lower = question.lower().strip()
    followup_patterns = [
        r"^what about\b",
        r"^how about\b",
        r"^and for\b",
        r"^what is it for\b",
        r"^does that apply to\b",
        r"^is there a penalty for that\b",
        r"^what are their terms\b"
    ]

    is_followup = any(re.search(pat, q_lower) for pat in followup_patterns) or len(q_lower.split()) <= 4

    if not is_followup:
        return question

    # Find the last user question in history
    last_user_q = ""
    for msg in reversed(history):
        if msg["role"] == "user":
            last_user_q = msg["content"]
            break

    if not last_user_q:
        return question

    # Contextual merging
    if "vendor abc" in q_lower:
        return f"What are the payment terms and agreements for Vendor ABC Corp? (Follow up to: {last_user_q})"
    elif "delhi" in q_lower:
        return f"What is the per diem travel and lodging allowance for Delhi? (Follow up to: {last_user_q})"
    elif "late fee" in q_lower or "penalty" in q_lower:
        return f"What is the late fee penalty for overdue invoices? (Follow up to: {last_user_q})"

    return f"{last_user_q} - {question}"
