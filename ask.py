import os
import sys
import argparse
import time
import json
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import (
    CHAT_MODEL,
    EMBED_MODEL,
    SIMILARITY_THRESHOLD,
    DB_BACKEND,
    RETRIEVAL_STRATEGY,
    GATE_MODE,
    PLANNER_MODE,
    AS_OF_DATE,
    run_startup_checks
)
from app.llm.generate import chat
from app.utils.formatters import format_inr

def print_invoice_table(invoices, limit: int = 10):
    """Prints a clean, formatted table of invoices with pagination."""
    if not invoices:
        return

    total = len(invoices)
    display_rows = invoices[:limit]

    print(f"\n--- MATCHING INVOICES TABLE (Showing {len(display_rows)} of {total}) ---")
    headers = f"{'INVOICE ID':<14} | {'VENDOR NAME':<26} | {'AMOUNT (INR)':<16} | {'STATUS':<10} | {'DUE DATE':<11} | {'PAID DATE':<10}"
    print("-" * len(headers))
    print(headers)
    print("-" * len(headers))

    for inv in display_rows:
        inv_id = str(inv.invoice_id)
        vendor = str(inv.vendor_name)[:25]
        amt = format_inr(inv.amount_inr)
        st = str(inv.status)
        due = str(inv.due_date)
        paid = str(inv.payment_date) if inv.payment_date else "-"
        print(f"{inv_id:<14} | {vendor:<26} | {amt:<16} | {st:<10} | {due:<11} | {paid:<10}")

    print("-" * len(headers))
    if total > limit:
        print(f"  [i] {total - limit} more invoice(s) omitted. Use '--limit {total}' to view all.")
    print()

def run_ask(
    question: str,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    router: str = "llm",
    retrieval_strategy: str = "hybrid",
    retrieval_mode: str = "always",
    keyword_bypass: bool = False,
    answer_gate: bool = True,
    gate_mode: str = "separate",
    use_reranker: bool = False,
    verify: bool = False,
    legacy_tools: bool = False,
    llm_data_answers: bool = False,
    top_k: int = 5,
    limit: int = 10,
    show_chunks: bool = False,
    embed_model: Optional[str] = None,
    planner: str = "auto",
    debug: bool = False
):
    selected_backend = (backend or DB_BACKEND).lower()
    target_model = model or CHAT_MODEL

    print("=" * 70)
    print(f"QUESTION: {question}")
    if debug:
        print(f"[DEBUG CONFIG] Backend: {selected_backend.upper()} | Search: {retrieval_strategy.upper()} | Mode: {retrieval_mode.upper()} | Gate: {answer_gate} ({gate_mode}) | Planner: {planner} | Legacy Tools: {legacy_tools} | LLM Data Answers: {llm_data_answers} | Model: {target_model}")
    print("=" * 70)

    # Execute chat response (runs Ollama query planner or tool loop)
    resp = chat(
        question=question,
        model=target_model,
        router=router,
        backend=selected_backend,
        retrieval_strategy=retrieval_strategy,
        retrieval_mode=retrieval_mode,
        keyword_bypass=keyword_bypass,
        answer_gate=answer_gate,
        gate_mode=gate_mode,
        use_reranker=use_reranker,
        verify=verify,
        legacy_tools=legacy_tools,
        llm_data_answers=llm_data_answers,
        embed_model=embed_model,
        planner=planner,
        debug=debug
    )

    # Print chunks retrieved from search_policies
    if show_chunks and resp.retrieved_chunks:
        q_label = f" for query '{resp.queries_run[-1]}'" if resp.queries_run else ""
        print(f"\n--- RETRIEVED CHUNKS ({len(resp.retrieved_chunks)}{q_label}) ---")
        for i, c in enumerate(resp.retrieved_chunks, 1):
            status_tag = " [ABOVE THRESHOLD]" if c.get("similarity", 0) >= SIMILARITY_THRESHOLD else " [BELOW THRESHOLD]"
            sim_score = c.get("similarity", 0.0)
            vr = f"#{c['vector_rank']}" if c.get("vector_rank") is not None else "None"
            kr = f"#{c['keyword_rank']}" if c.get("keyword_rank") is not None else "None"
            rrf_str = f" | RRF: {c['rrf_score']:.5f}" if "rrf_score" in c else ""
            print(f"[{i}] {c['source_file']}, Page {c['page']} / §{c['section']} (Vector Sim: {sim_score:.4f} | Vec Rank: {vr} | KW Rank: {kr}{rrf_str}){status_tag}")
            preview = c["content"].replace("\n", " ")[:120]
            print(f"    \"{preview}...\"")
        print("-" * 70)

    print("\n--- ANSWER ---")
    print(resp.answer)
    print("--- END ANSWER ---")

    if resp.sources:
        print("\n--- SOURCES ---")
        for s in resp.sources:
            print(f"  • {s}")

    # Python prints table of invoice rows after the summary
    if resp.invoices:
        print_invoice_table(resp.invoices, limit=limit)

    if debug:
        if resp.plan:
            print(f"\n[DEBUG PLAN] ({resp.plan_source}): {json.dumps(resp.plan)}")
        if resp.tool_results:
            print(f"[DEBUG TOOL RESULTS]: {json.dumps(resp.tool_results, default=str)}")
        if resp.retrieved_chunks:
            print(f"[DEBUG CHUNKS]: {json.dumps([{'source': c.get('source_file'), 'page': c.get('page'), 'section': c.get('section'), 'content': c.get('content')} for c in resp.retrieved_chunks])}")
        print(f"[DEBUG OLLAMA CALLS]: {resp.ollama_calls}")
        print(f"[DEBUG PROMPT TOKENS]: {resp.prompt_tokens}")
        if resp.stage_timings:
            print(f"[DEBUG STAGE TIMINGS]: {json.dumps(resp.stage_timings)}")
        print(f"\n[DEBUG METRICS] Backend: {selected_backend.upper()} | Route: {resp.route} | Total Latency: {resp.latency_ms}ms")

    print("=" * 70 + "\n")
    return resp


def main():
    parser = argparse.ArgumentParser(description="Finance RAG Assistant CLI with Local Ollama")
    parser.add_argument("question", nargs="*", help="Question to ask the finance assistant")
    parser.add_argument("--model", type=str, default=None, help=f"Ollama chat model override (default: {CHAT_MODEL})")
    parser.add_argument("--embed-model", type=str, default=None,
                        help="Ollama embedding model override (default: nomic-embed-text, choices: nomic-embed-text, bge-m3, mxbai-embed-large)")
    parser.add_argument("--backend", type=str, choices=["postgres", "sqlite"], default=None,
                        help="Vector database backend to use (default: postgres)")
    parser.add_argument("--retrieval", type=str, choices=["hybrid", "vector"], default=RETRIEVAL_STRATEGY,
                        help=f"Retrieval strategy: 'hybrid' (vector + FTS via RRF k=60, default: {RETRIEVAL_STRATEGY}) or 'vector'")
    parser.add_argument("--retrieval-mode", type=str, choices=["always", "model"], default="always",
                        help="Retrieval trigger mode: 'always' (default) or 'model'")
    parser.add_argument("--keyword-bypass", type=str, choices=["on", "off"], default="off",
                        help="Bypass SIMILARITY_THRESHOLD when keyword match exists in hybrid search (default: off)")
    parser.add_argument("--answer-gate", type=str, choices=["on", "off"], default="on",
                        help="Enable answerability gate check before generating final answer (default: on)")
    parser.add_argument("--gate", type=str, choices=["separate", "merged"], default=GATE_MODE,
                        help=f"Answerability gate mode: 'separate' or 'merged' (default: {GATE_MODE})")
    parser.add_argument("--planner", type=str, choices=["rules", "llm", "auto"], default=PLANNER_MODE,
                        help=f"Planner mode: 'auto' (rules first, LLM fallback, default: {PLANNER_MODE}), 'rules', or 'llm'")
    parser.add_argument("--legacy-tools", action="store_true",
                        help="Use legacy model-initiated tool-calling loop instead of query planner")
    parser.add_argument("--llm-data-answers", action="store_true",
                        help="Use LLM to generate answers for pure data queries instead of deterministic templates")
    parser.add_argument("--rerank", action="store_true", help="Apply pointwise LLM reranking to retrieved candidates")
    parser.add_argument("--verify", action="store_true", help="Run post-answer faithfulness verification audit")
    parser.add_argument("--limit", type=int, default=10, help="Max invoice rows to display in CLI table (default: 10)")
    parser.add_argument("--router", type=str, choices=["llm", "keyword"], default="llm",
                        help="Routing strategy: 'llm' (default) or legacy 'keyword'")
    parser.add_argument("--top-k", type=int, default=5, help="Number of policy chunks to retrieve (default: 5)")
    parser.add_argument("--show-chunks", action="store_true", help="Print actual retrieved chunks with similarity scores")
    parser.add_argument("--debug", action="store_true", help="Print backend, prompt tokens, timing, tool calls, and LLM call durations")
    parser.add_argument("--check", action="store_true", help="Run startup connectivity and model checks only")
    args = parser.parse_args()

    selected_backend = (args.backend or DB_BACKEND).lower()

    # Run startup check on launch
    checks = run_startup_checks(
        backend=selected_backend,
        chat_model=args.model,
        embed_model=args.embed_model,
        retrieval_mode=args.retrieval,
        gate_mode=args.gate,
        planner_mode=args.planner
    )
    if args.check:
        return

    # Check if Postgres was selected but offline
    if selected_backend == "postgres" and not checks.get("database_online", False):
        sep = "=" * 70
        sys.stderr.write(
            f"\n{sep}\n"
            "[FATAL ERROR] PostgreSQL with pgvector is unreachable!\n"
            "Silent fallback to SQLite has been removed.\n\n"
            "To start Postgres via Docker, run:\n"
            "    docker compose up -d db\n\n"
            "Or run explicitly with the SQLite backend:\n"
            "    python ask.py ... --backend sqlite\n"
            f"{sep}\n\n"
        )
        sys.exit(1)

    if args.question:
        q = " ".join(args.question)
        run_ask(
            question=q,
            model=args.model,
            backend=selected_backend,
            router=args.router,
            retrieval_strategy=args.retrieval,
            retrieval_mode=args.retrieval_mode,
            keyword_bypass=(args.keyword_bypass == "on"),
            answer_gate=(args.answer_gate == "on"),
            gate_mode=args.gate,
            use_reranker=args.rerank,
            verify=args.verify,
            legacy_tools=args.legacy_tools,
            llm_data_answers=args.llm_data_answers,
            top_k=args.top_k,
            limit=args.limit,
            show_chunks=args.show_chunks,
            embed_model=args.embed_model,
            planner=args.planner,
            debug=args.debug
        )
    else:
        print("Finance RAG Assistant CLI (type 'quit' or 'exit' to stop)")
        while True:
            try:
                user_q = input("\nAsk a question: ").strip()
                if not user_q or user_q.lower() in ("quit", "exit"):
                    break
                run_ask(
                    question=user_q,
                    model=args.model,
                    backend=selected_backend,
                    router=args.router,
                    retrieval_strategy=args.retrieval,
                    retrieval_mode=args.retrieval_mode,
                    keyword_bypass=(args.keyword_bypass == "on"),
                    answer_gate=(args.answer_gate == "on"),
                    gate_mode=args.gate,
                    use_reranker=args.rerank,
                    verify=args.verify,
                    legacy_tools=args.legacy_tools,
                    llm_data_answers=args.llm_data_answers,
                    top_k=args.top_k,
                    limit=args.limit,
                    show_chunks=args.show_chunks,
                    embed_model=args.embed_model,
                    planner=args.planner,
                    debug=args.debug
                )
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
