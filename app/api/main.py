import os
import sys
import time
import shutil
import logging
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.db import get_connection, init_db
from app.llm.schemas import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
    UploadResponse,
    QueryRequest,
    QueryResponse
)
from app.llm.generate import chat, answer_query
from app.llm.history import get_session_history, add_to_session, rewrite_followup_query
from app.retrieval.search import search_chunks
from app.ingestion.parse import extract_pages, clean_text
from app.ingestion.chunk import chunk_page_text
from app.ingestion.embed import get_embeddings
from app.tools.invoice_queries import query_invoices, get_invoice_metrics

# -------------------------------------------------------------
# Setup Logging
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("finance_rag")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")
os.makedirs(POLICIES_DIR, exist_ok=True)

app = FastAPI(
    title="Finance RAG & Invoice Assistant",
    description="Dual-path RAG and structured table query engine with tool calling.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} ({duration:.2f}ms)")
    return response

# -------------------------------------------------------------
# Core Phase 8 API Endpoints
# -------------------------------------------------------------

from app.config import run_startup_checks, OLLAMA_HOST, CHAT_MODEL, EMBED_MODEL, get_ollama_client

@app.on_event("startup")
def on_startup():
    logger.info("Starting Finance RAG API Server...")
    run_startup_checks()

@app.get("/health")
def health_check():
    """Checks database connectivity, table statistics, and Ollama server status."""
    ollama_ok = False
    models_available = []
    try:
        client = get_ollama_client()
        m_list = client.list()
        ollama_ok = True
        models_available = [m.model for m in m_list.models]
    except Exception as e:
        logger.warning(f"Ollama health ping failed: {e}")

    try:
        conn, engine = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks;")
        chunks_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM invoices;")
        invoices_count = cur.fetchone()[0]
        conn.close()

        return {
            "status": "healthy" if ollama_ok else "degraded",
            "ollama_host": OLLAMA_HOST,
            "ollama_online": ollama_ok,
            "models_ready": models_available,
            "database_engine": engine,
            "chunks_count": chunks_count,
            "invoices_count": invoices_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database connectivity failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Primary chat endpoint with conversational history rewriting.
    Routes queries to policy search, invoice lookup, or hybrid execution.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = req.session_id or "default"
    history = get_session_history(session_id)

    # Rewrite follow-up query if contextual
    rewritten_q = rewrite_followup_query(req.question, history)

    t0 = time.time()
    try:
        resp = chat(rewritten_q)
        latency = round((time.time() - t0) * 1000, 2)
        resp.latency_ms = latency

        # Save turn to conversation history
        add_to_session(session_id, "user", req.question)
        add_to_session(session_id, "assistant", resp.answer)

        logger.info(
            f"Session: {session_id} | Raw Q: '{req.question}' | Rewritten: '{rewritten_q}' | "
            f"Route: {resp.route} | Sources: {len(resp.sources)} | Latency: {latency}ms"
        )
        return resp
    except Exception as e:
        logger.error(f"Chat processing error for '{req.question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    """
    Streaming chat endpoint yielding token chunks via Server-Sent Events (SSE).
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = req.session_id or "default"
    history = get_session_history(session_id)
    rewritten_q = rewrite_followup_query(req.question, history)

    def event_stream():
        resp = chat(rewritten_q)
        # Yield metadata event
        yield f"event: metadata\ndata: {json.dumps({'route': resp.route, 'sources': resp.sources})}\n\n"

        # Stream answer tokens with minor delay simulating generator
        words = resp.answer.split(" ")
        for word in words:
            yield f"data: {word} \n\n"
            time.sleep(0.01)

        # Save to history
        add_to_session(session_id, "user", req.question)
        add_to_session(session_id, "assistant", resp.answer)
        yield "event: end\ndata: [DONE]\n\n"

    import json
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/search", response_model=SearchResponse)
def raw_search_endpoint(req: SearchRequest):
    """
    Raw vector retrieval endpoint with no LLM synthesis, designed for debugging and evaluation.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    t0 = time.time()
    try:
        raw_hits = search_chunks(req.query, doc_type=req.doc_type, limit=req.limit)
        latency = (time.time() - t0) * 1000
        logger.info(f"Raw Search: '{req.query}' | Hits: {len(raw_hits)} | Latency: {latency:.2f}ms")

        hits = [
            SearchHit(
                content=h["content"],
                source_file=h["source_file"],
                page=h["page"],
                section=h["section"],
                similarity=h["similarity"]
            )
            for h in raw_hits
        ]
        return SearchResponse(query=req.query, results=hits, count=len(hits))
    except Exception as e:
        logger.error(f"Search retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a new policy PDF, runs page extraction, text cleaning, tiktoken chunking,
    and inserts embeddings into the database.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    dest_path = os.path.join(POLICIES_DIR, file.filename)
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest document
        pages = extract_pages(dest_path)
        new_chunks = []
        for page_num, raw_text in pages:
            cleaned = clean_text(raw_text)
            chunks = chunk_page_text(
                source_file=file.filename,
                page=page_num,
                cleaned_text=cleaned,
                target_chunk_tokens=500,
                overlap_tokens=50
            )
            new_chunks.extend(chunks)

        if new_chunks:
            texts = [c["content"] for c in new_chunks]
            embs = get_embeddings(texts)

            conn, engine = get_connection()
            cur = conn.cursor()

            if engine == "postgres":
                insert_sql = """
                    INSERT INTO chunks (source_file, page, section, doc_type, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """
                data = [
                    (c["source_file"], c["page"], c["section"], c["doc_type"], c["content"], embs[i])
                    for i, c in enumerate(new_chunks)
                ]
                cur.executemany(insert_sql, data)
            else:
                import json
                insert_sql = """
                    INSERT INTO chunks (source_file, page, section, doc_type, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?);
                """
                data = [
                    (c["source_file"], c["page"], c["section"], c["doc_type"], c["content"], json.dumps(embs[i]))
                    for i, c in enumerate(new_chunks)
                ]
                cur.executemany(insert_sql, data)

            conn.commit()
            conn.close()

        logger.info(f"Uploaded and ingested {file.filename}: {len(new_chunks)} chunks created.")
        return UploadResponse(
            filename=file.filename,
            chunks_created=len(new_chunks),
            status="success"
        )
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document upload and ingestion failed: {str(e)}")

# Legacy compatibility endpoints
@app.post("/api/query", response_model=QueryResponse)
def handle_query_legacy(req: QueryRequest):
    return answer_query(req.question)

@app.get("/api/invoices")
def list_invoices(
    status: Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    vendor: Optional[str] = Query(None),
    limit: int = Query(50)
):
    return query_invoices(status=status, min_amount=min_amount, max_amount=max_amount, vendor=vendor, limit=limit)

@app.get("/api/metrics")
def get_metrics():
    return get_invoice_metrics()

# -------------------------------------------------------------
# Modern Interactive UI
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Finance RAG Assistant & Router</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --bg-card: #182234;
            --text-primary: #F8FAFC;
            --text-muted: #94A3B8;
            --accent: #38BDF8;
            --accent-glow: rgba(56, 189, 248, 0.15);
            --border: #334155;
            --badge-policy: #10B981;
            --badge-data: #F59E0B;
            --badge-hybrid: #8B5CF6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 24px;
        }
        .container { max-width: 1050px; margin: 0 auto; }
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        .logo-group h1 { font-size: 22px; font-weight: 700; color: #fff; }
        .logo-group p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
        .badge-status {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34D399;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }
        .input-row { display: flex; gap: 10px; }
        input[type="text"] {
            flex: 1;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            color: #fff;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }
        button {
            background: linear-gradient(135deg, #0284C7, #0369A1);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 0 24px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }

        .sample-queries {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }
        .sample-pill {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .sample-pill:hover {
            color: var(--text-primary);
            border-color: var(--accent);
        }

        .result-box { display: none; }
        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .badge.POLICY_RAG { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
        .badge.INVOICE_DATA { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
        .badge.HYBRID, .badge.TOOL_CALLS { background: rgba(139, 92, 246, 0.2); color: #C084FC; border: 1px solid #8B5CF6; }

        .answer-content {
            font-size: 14px;
            line-height: 1.6;
            color: #E2E8F0;
            white-space: pre-wrap;
            margin-bottom: 16px;
        }
        .citations-section {
            border-top: 1px solid var(--border);
            padding-top: 12px;
            margin-top: 12px;
        }
        .citations-section h4 {
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .citation-tag {
            display: inline-block;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: var(--accent);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            margin-right: 6px;
            margin-bottom: 6px;
        }

        .table-wrapper {
            margin-top: 16px;
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th { background: var(--bg-secondary); color: var(--text-muted); font-weight: 600; }
        tr:hover { background: rgba(255, 255, 255, 0.02); }
        .status-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .status-Paid { background: rgba(16, 185, 129, 0.2); color: #34D399; }
        .status-Pending { background: rgba(56, 189, 248, 0.2); color: #38BDF8; }
        .status-Overdue { background: rgba(239, 68, 68, 0.2); color: #F87171; }
        
        .upload-section {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 10px;
        }
        .file-input {
            color: var(--text-muted);
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-group">
                <h1>Finance RAG Assistant & Router</h1>
                <p>Phases 5-8: Numbered Citations, Parameterized SQL Tools, Router, & FastAPI</p>
            </div>
            <div class="badge-status" id="health-status">Checking API...</div>
        </header>

        <div class="card">
            <div class="input-row">
                <input type="text" id="question-input" placeholder="Ask a question (e.g. 'Is INV-2024-001 overdue, and what is the late fee?')..." />
                <button id="ask-btn" onclick="submitQuery()">Ask</button>
            </div>
            <div class="sample-queries">
                <span class="sample-pill" onclick="setQuery('What is our payment policy?')">📋 Payment policy</span>
                <span class="sample-pill" onclick="setQuery('Show invoices above ₹1 lakh that are unpaid')">💰 Unpaid &gt; ₹1 Lakh</span>
                <span class="sample-pill" onclick="setQuery('Is INV-2024-001 overdue, and what is the late fee?')">🔀 Mixed: INV-2024-001 + Late fee</span>
                <span class="sample-pill" onclick="setQuery('What are the payment terms for Vendor ABC Corp?')">🏢 Vendor ABC terms</span>
                <span class="sample-pill" onclick="setQuery('List overdue invoices for IT department')">💻 IT overdue</span>
                <span class="sample-pill" onclick="setQuery('What is our policy on cryptocurrency payments?')">🛡️ Anti-hallucination test</span>
            </div>
        </div>

        <div class="card result-box" id="result-box">
            <div class="result-header">
                <h3 id="result-title">Response</h3>
                <div>
                    <span id="route-badge" class="badge"></span>
                    <span id="latency-tag" style="font-size: 12px; color: var(--text-muted); margin-left: 8px;"></span>
                </div>
            </div>
            <div class="answer-content" id="answer-content"></div>
            <div id="citations-container" class="citations-section" style="display: none;">
                <h4>Numbered Citations</h4>
                <div id="citations-list"></div>
            </div>
            <div id="table-container" class="table-wrapper" style="display: none;">
                <table>
                    <thead>
                        <tr>
                            <th>Invoice ID</th>
                            <th>Vendor</th>
                            <th>Date</th>
                            <th>Due Date</th>
                            <th>Amount (INR)</th>
                            <th>Status</th>
                            <th>Dept</th>
                        </tr>
                    </thead>
                    <tbody id="invoice-rows"></tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <h3 style="font-size: 15px; margin-bottom: 8px;">Upload New Policy PDF (POST /documents/upload)</h3>
            <div class="upload-section">
                <input type="file" id="pdf-file-input" accept=".pdf" class="file-input" />
                <button onclick="uploadDocument()" style="padding: 6px 16px; font-size: 13px;">Upload & Ingest</button>
                <span id="upload-status" style="font-size: 13px; color: var(--accent);"></span>
            </div>
        </div>
    </div>

    <script>
        fetch('/health')
            .then(r => r.json())
            .then(data => {
                document.getElementById('health-status').innerText = 
                    `DB: ${data.database_engine.toUpperCase()} (${data.chunks_count} chunks, ${data.invoices_count} invoices)`;
            })
            .catch(() => {
                document.getElementById('health-status').innerText = 'Offline';
            });

        function setQuery(text) {
            document.getElementById('question-input').value = text;
            submitQuery();
        }

        async function submitQuery() {
            const input = document.getElementById('question-input');
            const btn = document.getElementById('ask-btn');
            const q = input.value.trim();
            if (!q) return;

            btn.disabled = true;
            btn.innerText = 'Routing...';

            const resultBox = document.getElementById('result-box');
            resultBox.style.display = 'block';
            document.getElementById('answer-content').innerText = 'Processing query...';
            document.getElementById('citations-container').style.display = 'none';
            document.getElementById('table-container').style.display = 'none';

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q })
                });
                const data = await res.json();

                const badge = document.getElementById('route-badge');
                badge.className = `badge ${data.route || 'HYBRID'}`;
                badge.innerText = data.route || 'ROUTER';

                if (data.latency_ms) {
                    document.getElementById('latency-tag').innerText = `${data.latency_ms}ms`;
                }

                document.getElementById('answer-content').innerText = data.answer;

                if (data.sources && data.sources.length > 0) {
                    const citDiv = document.getElementById('citations-list');
                    citDiv.innerHTML = '';
                    data.sources.forEach(src => {
                        const span = document.createElement('span');
                        span.className = 'citation-tag';
                        span.innerText = src;
                        citDiv.appendChild(span);
                    });
                    document.getElementById('citations-container').style.display = 'block';
                }

                if (data.invoices && data.invoices.length > 0) {
                    const tbody = document.getElementById('invoice-rows');
                    tbody.innerHTML = '';
                    data.invoices.forEach(inv => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td><b>${inv.invoice_id}</b></td>
                            <td>${inv.vendor_name}</td>
                            <td>${inv.invoice_date}</td>
                            <td>${inv.due_date}</td>
                            <td>₹${Number(inv.amount_inr).toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
                            <td><span class="status-badge status-${inv.status}">${inv.status}</span></td>
                            <td>${inv.department || '-'}</td>
                        `;
                        tbody.appendChild(tr);
                    });
                    document.getElementById('table-container').style.display = 'block';
                }
            } catch (err) {
                document.getElementById('answer-content').innerText = 'Error: ' + err.message;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Ask';
            }
        }

        async function uploadDocument() {
            const fileInput = document.getElementById('pdf-file-input');
            const status = document.getElementById('upload-status');
            if (!fileInput.files.length) {
                status.innerText = 'Please select a PDF file first.';
                return;
            }
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            status.innerText = 'Uploading and ingesting...';
            try {
                const res = await fetch('/documents/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                status.innerText = `Ingested ${data.filename}: created ${data.chunks_created} chunks.`;
            } catch (e) {
                status.innerText = 'Upload failed: ' + e.message;
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
