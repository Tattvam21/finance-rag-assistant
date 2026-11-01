#!/usr/bin/env python3
"""
scripts/build_readme.py
Generates README.md dynamically with evaluation numbers extracted directly from the latest
eval/results JSON files.
"""

import os
import json
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"]
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    pass_pct = 100 * passed / total if total else 0.0
    lats = [r["latency_s"] for r in rows]
    mean_lat = statistics.mean(lats) if lats else 0.0
    p95_lat = sorted(lats)[min(len(lats)-1, int(0.95*len(lats)))] if lats else 0.0
    calls = [r.get("ollama_calls", 0) for r in rows if "ollama_calls" in r]
    mean_calls = statistics.mean(calls) if calls else 0.0

    cats = {}
    for r in rows:
        c = r["category"]
        if c not in cats:
            cats[c] = {"total": 0, "passed": 0}
        cats[c]["total"] += 1
        if r["passed"]:
            cats[c]["passed"] += 1

    return {
        "file": os.path.basename(path),
        "total": total,
        "passed": passed,
        "pass_pct": pass_pct,
        "mean_lat": mean_lat,
        "p95_lat": p95_lat,
        "mean_calls": mean_calls,
        "cats": cats
    }

def main():
    dev_path = os.path.join(PROJECT_ROOT, "eval/results/20261031_124054_cfg_e_postgres_dev.json")
    held_path = os.path.join(PROJECT_ROOT, "eval/results/20261031_124417_cfg_e_postgres_heldout.json")

    dev = load_results(dev_path)
    held = load_results(held_path)

    dev_table_rows = "\n".join([
        f"| **{cat}** | {st['passed']}/{st['total']} | {100*st['passed']/st['total']:.1f}% |"
        for cat, st in sorted(dev["cats"].items())
    ])

    held_table_rows = "\n".join([
        f"| **{cat}** | {st['passed']}/{st['total']} | {100*st['passed']/st['total']:.1f}% |"
        for cat, st in sorted(held["cats"].items())
    ])

    readme_content = f"""# Finance RAG & Invoice Assistant

A production-grade, hybrid financial assistant that bridges unstructured company policies and structured relational invoice databases through an **intelligent dual-path planner** with **deterministic claim guards**.

---

## Architecture

```mermaid
flowchart TD
    User([User Question]) --> FastAPIServer[FastAPI Server / CLI]
    FastAPIServer --> QueryPlanner{{Query Planner / Tool Classifier}}

    %% Path 1: Policy Vector RAG
    QueryPlanner -->|Policy Questions| HybridSearch[pgvector + BM25 Hybrid Retrieval - Top 5]
    HybridSearch --> PolicyContext[Extracted Policy Context with Exact Inline Citations]

    %% Path 2: Structured SQL Data
    QueryPlanner -->|Invoice Inquiries| SQLTools[Parameterized Safe SQL Query / Late Fee Tool]
    SQLTools --> PostgresDB[(PostgreSQL / pgvector)]
    PostgresDB --> InvoiceRows[Structured Invoice Records & Status Aggregates]

    %% Synthesis & Verification
    PolicyContext --> LLMSynthesis[Ollama Qwen 8B LLM Synthesis]
    InvoiceRows --> LLMSynthesis

    LLMSynthesis --> DeterministicGuards{{Deterministic Claim Guards}}
    DeterministicGuards -->|Numeric Verification| NumericGuard[Value, Unit, and Multi-Currency Tolerance Check]
    DeterministicGuards -->|Missing Condition| ConditionAppend[Verbatim Policy Clause Append]
    DeterministicGuards -->|Data Contradiction| PureDataGuard[Deterministic Pure-Data Template Replacement]
    DeterministicGuards -->|Unanswerable Gate| RefusalGate[Strict Policy Absence Refusal]

    NumericGuard --> FinalAnswer([Verified Answer + Citations + Invoice Tables])
    ConditionAppend --> FinalAnswer
    PureDataGuard --> FinalAnswer
    RefusalGate --> FinalAnswer

    FinalAnswer --> SSEStreaming[SSE Streaming /chat/stream]
    FinalAnswer --> JSONResponse[JSON REST API /chat]
```

---

## Features

1. **Dual-Path Architecture**:
   - **Policy Documents**: Parsed from PDFs, chunked with section hierarchy, embedded with `bge-m3` (1024-dim), and searched using hybrid vector cosine distance + full-text search merged via Reciprocal Rank Fusion (RRF, $k=60$).
   - **Relational Invoices**: Structured queries, invoice status breakdowns, and late fee calculations execute directly against PostgreSQL with parameterized queries.
2. **Deterministic Claim Guards (Zero Extra LLM Calls)**:
   - **Numeric Token Grounding**: Enforces strict value-unit pairings (percentages, day counts, lakhs/crores).
   - **Data Claim Guard**: Verifies counts (unpaid, overdue, pending), sums, and amounts against SQL ground truth; deterministic data sentences overwrite hallucinations without re-calling the LLM.
   - **Gap & Condition Handling**: Distinguishes between unanswerable questions (refusal) and policy omissions (gap overrides). Verbatim conditions are appended when missing.
3. **Containerized REST API & SSE Streaming**:
   - Packaged with Docker Compose (`api` on port 8000, `db` with PostgreSQL 16 pgvector on port 5432).
   - Ollama runs natively on the Mac host, reachable from inside Docker via `http://host.docker.internal:11434`.

---

## Project Structure

```
.
├── Dockerfile                  # API container (Python 3.13-slim)
├── docker-compose.yml          # Postgres (pgvector) + FastAPI orchestration
├── requirements.txt            # Python dependencies
├── .env.example                # Template with non-secret defaults & blanked credentials
├── ask.py                      # CLI entrypoint for querying the assistant
├── app/
│   ├── config.py               # Runtime configuration & startup health checks
│   ├── db.py                   # PostgreSQL connection management & pgvector init
│   ├── api/
│   │   ├── main.py             # FastAPI server (/chat, /chat/stream, /search, /documents/upload, /health)
│   │   └── index.html          # Interactive Web UI
│   ├── ingestion/              # PDF parsing, semantic chunking, and embedding
│   ├── retrieval/              # Hybrid vector + lexical search with RRF reranking
│   ├── tools/                  # Parameterized invoice and late fee calculations
│   ├── llm/                    # Query planner, prompt synthesis, and conversation history
│   └── utils/                  # Deterministic guards, formatters, and numeric helpers
├── data/
│   ├── invoices.csv            # Relational invoice records
│   └── policies/               # Financial policy PDF documents
├── eval/                       # Evaluation suites, benchmark questions, and results
├── scripts/                    # Ingestion, data generation, and benchmark scripts
└── tests/                      # Pytest suite (API endpoints, guards, numeric handling)
```

---

## Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Native [Ollama](https://ollama.com) running on macOS with required models:
  ```bash
  ollama pull qwen3:8b
  ollama pull bge-m3
  ```

### 2. Environment Setup
```bash
cp .env.example .env
```
*(Update `POSTGRES_PASSWORD` in `.env` if desired; non-secret defaults are pre-configured.)*

### 3. Launch with Docker Compose
```bash
docker compose up -d
```
Verify container statuses:
```bash
docker compose ps
```
*(Note: `docker-compose.yml` automatically overrides `DATABASE_URL` to route to the `db` service container and `OLLAMA_HOST` to `http://host.docker.internal:11434`, while native host runs like `python ask.py` continue using `localhost` from `.env`.)*

### 4. Load Policy Documents & Invoices
To populate PostgreSQL with the policy chunks and invoices, run:
```bash
docker compose exec api python scripts/ingest_data.py
```
*(Or locally: `python scripts/ingest_data.py`)*

### 5. Run CLI Assistant (Local Python)
```bash
python ask.py "What is the payment period?"
python ask.py "What are the payment terms for Vendor ABC Corp, and how many of its invoices are unpaid?"
```

---

## API Documentation & Examples

The FastAPI server provides OpenAPI documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### 1. Health Check (`GET /health`)
Verifies PostgreSQL connectivity, Ollama reachability, model availability, and row counts:
```bash
curl -s http://localhost:8000/health
```
```json
{{
  "status": "healthy",
  "database_engine": "postgres",
  "database_connected": true,
  "ollama_host": "http://host.docker.internal:11434",
  "ollama_online": true,
  "models": {{
    "chat_model": "qwen3:8b",
    "embed_model": "bge-m3",
    "chat_model_ready": true,
    "embed_model_ready": true
  }},
  "chunks_count": 18,
  "invoices_count": 90
}}
```

### 2. Raw Vector Retrieval (`POST /search`)
Direct hybrid retrieval without calling the LLM:
```bash
curl -s -X POST http://localhost:8000/search \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "What is the payment period?", "top_k": 2}}'
```

### 3. Primary Chat (`POST /chat`)
Executes the complete dual-path query planner, tool invocation, and deterministic guard pipeline:
```bash
curl -s -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "What are the payment terms for Vendor ABC Corp, and how many of its invoices are unpaid?"}}'
```

### 4. Server-Sent Events Streaming (`POST /chat/stream`)
Real-time status updates followed by answer token chunks and citations:
```bash
curl -N -s -X POST http://localhost:8000/chat/stream \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "What is the payment period?"}}'
```
**Streaming Guard Tradeoff**:
Because our deterministic correctness guards (numeric grounding, unanswerable gate, condition appending, data claim guard) inspect the complete synthesized answer against SQL ground truth and cited chunk numbers, `POST /chat/stream` emits status events (`planning`, `retrieving`, `generating`), runs the verification pipeline, and then streams the **verified final answer** in tokens followed by a final event containing verified citations. This prevents partial streaming of unverified hallucinations.

### 5. Document Ingestion (`POST /documents/upload`)
Uploads and indexes a new policy PDF or invoice CSV:
```bash
curl -s -X POST http://localhost:8000/documents/upload \\
  -F "file=@data/policies/payment_policy.pdf"
```

---

## Evaluation Benchmark

The system is evaluated using the automated evaluation harness (`eval/run_eval.py`) across ground-truth suites on PostgreSQL under Config E (`qwen3:8b`, `bge-m3`, hybrid retrieval, auto planner, merged gate). The statistics below are generated directly from the latest evaluation results files:

### Dev Benchmark (`{dev["file"]}`)
- **Pass Rate**: **{dev["passed"]}/{dev["total"]} ({dev["pass_pct"]:.1f}%)**
- **Latency**: Mean {dev["mean_lat"]:.1f}s | p95 {dev["p95_lat"]:.1f}s
- **Ollama Calls per Question**: {dev["mean_calls"]:.2f}

| Category | Passed / Total | Pass Rate |
| :--- | :--- | :--- |
{dev_table_rows}

### Held-out Benchmark (`{held["file"]}`)
- **Pass Rate**: **{held["passed"]}/{held["total"]} ({held["pass_pct"]:.1f}%)**
- **Latency**: Mean {held["mean_lat"]:.1f}s | p95 {held["p95_lat"]:.1f}s
- **Ollama Calls per Question**: {held["mean_calls"]:.2f}

| Category | Passed / Total | Pass Rate |
| :--- | :--- | :--- |
{held_table_rows}

---

## Limitations

1. **Synthetic Dataset Scope**:
   - The policy documents cover 18 distinct sections across 5 corporate finance PDFs, and the database contains 90 synthetic invoices. Real-world corporate deployments feature thousands of documents and millions of ledger transactions that require sharded vector indices and partition pruning.
2. **Local 8B Model Arithmetic & Phrasing Variations**:
   - Small local LLMs (e.g. `qwen3:8b`) can exhibit stochastic phrasing variations or conflate sub-fields (such as mistaking "4 overdue invoices" for "total unpaid invoices" prior to guard intervention). The system relies on deterministic post-generation guards to enforce accuracy on numerical counts and financial totals.
3. **Automated LLM Judge Reliability**:
   - Standard automated LLM judges were found to be brittle and inconsistent when auditing fine-grained numerical equivalence (e.g. Indian numbering grouping vs. Western notation). As a result, faithfulness verification was audited manually across 38 representative answer pairs to ensure grounding.
4. **Deterministic Guard Boundaries**:
   - The guards use deterministic regex parsers and value-unit extractors designed for financial terminology (currencies, percentages, day thresholds). Phrasings outside these patterns may bypass specific check heuristics.
5. **Development Set Iteration**:
   - The dev (40 questions) and held-out (22 questions) benchmark sets were utilized during prompt engineering and pipeline debugging. True out-of-distribution evaluation requires testing on unseen policy domains.
"""

    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content.strip() + "\n")
    print(f"Successfully generated README.md at {readme_path}")

if __name__ == "__main__":
    main()

