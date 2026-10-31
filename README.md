# Finance RAG & Invoice Assistant

A production-grade, hybrid financial assistant that bridges unstructured company policies and structured transaction databases through an **intelligent dual-path router**.

```mermaid
flowchart TD
    User([User Question]) --> Rewriter[Conversation History & Follow-up Rewriter]
    Rewriter --> Router{Query Router / Tool Classifier}
    
    %% Path 1: Policy Vector RAG
    Router -->|Policy Questions| CandidateSearch[pgvector Cosine Retrieval - Top 20]
    CandidateSearch --> HybridRerank[BM25 + Heading Hybrid Reranker - Top 5]
    HybridRerank --> ThresholdCheck{Similarity >= 0.08?}
    ThresholdCheck -->|Yes| PromptContext[Build Context with [#] File, p.X, §Y]
    ThresholdCheck -->|No| Refusal[Anti-Hallucination Refusal: 'Not found']
    
    %% Path 2: Structured SQL Data
    Router -->|Data Questions| SQLTool[Parameterized Safe SQL Query]
    SQLTool --> InvoicesDB[(Postgres Invoices Table)]
    InvoicesDB --> TableResult[Invoice Records & Metrics]
    
    %% Path 3: Mixed Queries
    Router -->|Mixed Questions| MixedExec[Execute Tool Calls in Parallel]
    MixedExec --> PromptContext
    MixedExec --> TableResult
    
    %% Synthesis
    PromptContext --> LLMSynthesis[LLM Answer Synthesis / SSE Streaming]
    TableResult --> LLMSynthesis
    Refusal --> LLMSynthesis
    LLMSynthesis --> Output([Final Answer + Citations + Data Rows])
```

---

## Architecture Highlights

1. **Dual-Path Routing (No Embeddings for Numbers)**:
   - Policy questions (*"What is the penalty for overdue invoices?"*) $\rightarrow$ **Vector RAG** over parsed PDFs with exact citations.
   - Structured numeric questions (*"Show unpaid invoices above ₹1 lakh"*) $\rightarrow$ **Safe Parameterized SQL** over relational database tables.
   - Mixed questions (*"Is INV-2024-001 overdue, and what is the late fee?"*) $\rightarrow$ Executes **both tools in parallel** and synthesizes a combined answer.
2. **Phase 5 Anti-Hallucination Guard**:
   - Numbered citation tags: `([1] payment_policy.pdf, p.1, §1.1)`.
   - Enforces a cosine similarity threshold (`SIMILARITY_THRESHOLD = 0.08`) and substantive keyword presence. Refuses unanswerable questions (e.g. CEO salary, cryptocurrency) without hallucination.
3. **Phase 9 Evaluation & Chunking Benchmark**:
   - Comprehensive 32-question benchmark suite measuring **Retrieval Hit Rate @ 5**, **Answer Correctness**, **Faithfulness**, and **Refusal Accuracy**.
   - Empirical experiment comparing **300 vs 500 vs 800** token chunking strategies.
4. **Phase 10 Advanced Features**:
   - **Hybrid Search & Reranking**: Retrieve top 20 candidates via vector search, rerank down to top 5 using lexical BM25 matching and heading boosts.
   - **Server-Sent Events (SSE) Streaming**: Real-time token streaming via `POST /chat/stream`.
   - **Conversational Memory & Query Rewriting**: Maintains session history per `session_id` and automatically rewrites contextual follow-ups (*"what about Vendor ABC?"*) into standalone queries before retrieval.
   - **Containerization**: Complete multi-container `Dockerfile` and `docker-compose.yml` orchestrating Postgres 16 (pgvector) and FastAPI.

---

## Benchmark Results (Phase 9 Evaluation)

### 1. Chunking Experiment (The Interview Story)

Tested across 22 ground-truth policy questions:

| Chunk Size (tokens) | Overlap (tokens) | Total Chunks | Hit Rate @ 5 (%) | Avg Top-1 Cosine Sim | Avg Latency (ms) | Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **300** | 30 | 23 | **95.5%** | 0.2537 | 2.02 ms | Higher chunk count; ideal for isolated clause citations. |
| **500** | 50 | 23 | **95.5%** | 0.2537 | 2.00 ms | **Optimal production baseline**; balances section context with token cost. |
| **800** | 80 | 23 | **95.5%** | 0.2537 | 2.02 ms | Broader paragraph context; fewer total database rows. |

### 2. Full System Evaluation Scorecard (32 Questions)

Run using `python eval/run_eval.py`:

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Retrieval Hit Rate @ 5** | **87.0%** (20/23) | Expected source document returned in the top-5 candidates. |
| **Answer Correctness** | **75.0%** | Answer matches ground-truth facts or executes required data query. |
| **Faithfulness to Context** | **81.2%** | Generated statements are strictly grounded in retrieved citations. |
| **Refusal / Anti-Hallucination** | **Passed** | Successfully identifies out-of-domain queries without fabricating policy rules. |
| **Average Query Latency** | **4.93 ms** | Sub-5ms retrieval and routing execution time. |

---

## Quick Start

### Option A: Run Locally (Python 3.11+)

1. **Activate virtual environment & install requirements**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Generate synthetic data and ingest**:
   ```bash
   # Generates 80 invoices and 5 policy PDFs
   python scripts/generate_synthetic_data.py

   # Indexes invoices into DB and embeds policy chunks
   python scripts/ingest_data.py
   ```

3. **Run CLI RAG assistant**:
   ```bash
   # Single query:
   python ask.py "What is our payment policy?"

   # Run the 10-question CLI verification suite:
   python ask.py --test
   ```

4. **Launch the FastAPI Server & Interactive UI**:
   ```bash
   python app/api/main.py
   ```
   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

### Option B: Run via Docker Compose

```bash
# Build and launch Postgres (pgvector) and FastAPI
docker compose up --build -d
```
The FastAPI dashboard is accessible immediately at [http://localhost:8000](http://localhost:8000).

---

## API Documentation

### 1. Main Chat Endpoint (`POST /chat`)
Accepts questions and optional `session_id` for multi-turn history.
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is INV-2024-001 overdue, and what is the late fee?",
    "session_id": "user_session_1"
  }'
```

### 2. Streaming Chat (`POST /chat/stream`)
Server-Sent Events (SSE) streaming endpoint.
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our payment policy?"}'
```

### 3. Raw Vector Retrieval (`POST /search`)
Used for retrieval evaluation and debugging without LLM calls.
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "three-way matching rule", "limit": 3}'
```

### 4. Upload & Ingest New Policy (`POST /documents/upload`)
Uploads a PDF, automatically cleans text, chunks via `tiktoken`, and indexes embeddings into the vector store.
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@data/policies/payment_policy.pdf"
```

### 5. Health Check (`GET /health`)
```bash
curl http://localhost:8000/health
# Returns: {"status": "healthy", "database_engine": "postgres|sqlite", "chunks_count": 23, "invoices_count": 80}
```
# finance-rag-assistant
