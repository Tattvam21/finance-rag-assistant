import os
import sys
import time
import json
import shutil
import logging
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.db import get_connection, init_db
from app.config import (
    OLLAMA_HOST,
    CHAT_MODEL,
    EMBED_MODEL,
    RETRIEVAL_STRATEGY,
    GATE_MODE,
    PLANNER_MODE,
    DB_BACKEND,
    run_startup_checks,
    get_ollama_client
)
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
from app.llm.generate import chat
from app.llm.history import get_session_history, add_to_session, rewrite_followup_query
from app.retrieval.search import search_chunks
from app.ingestion.parse import extract_pages, clean_text
from app.ingestion.chunk import chunk_page_text
from app.ingestion.embed import get_embeddings, get_embedder_spec
from app.tools.invoice_queries import query_invoices

# -------------------------------------------------------------
# Setup Logging
# -------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("finance_rag.api")

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

# Standard JSON error response handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc), "status_code": 500}
    )

@app.on_event("startup")
def on_startup():
    logger.info("Starting Finance RAG API Server...")
    try:
        run_startup_checks()
    except Exception as e:
        logger.warning(f"Startup check warning: {e}")

# -------------------------------------------------------------
# Core API Endpoints
# -------------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Checks database connectivity, table statistics, Ollama server status,
    and availability of required chat and embedding models.
    """
    ollama_ok = False
    models_available = []
    chat_model_ready = False
    embed_model_ready = False

    try:
        client = get_ollama_client()
        m_list = client.list()
        ollama_ok = True
        models_available = [m.model for m in m_list.models]
        chat_model_ready = any(CHAT_MODEL in m for m in models_available)
        embed_model_ready = any(EMBED_MODEL in m for m in models_available)
    except Exception as e:
        logger.warning(f"Ollama health check ping failed: {e}")

    try:
        conn, engine = get_connection(backend=DB_BACKEND)
        cur = conn.cursor()

        spec = get_embedder_spec(EMBED_MODEL)
        table_name = spec.get("table", "chunks_bge")
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table_name};")
            chunks_count = cur.fetchone()[0]
        except Exception:
            if engine == "postgres":
                conn.rollback()
            cur.execute("SELECT COUNT(*) FROM chunks;")
            chunks_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM invoices;")
        invoices_count = cur.fetchone()[0]
        conn.close()
        postgres_ok = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "unreachable",
                "error": str(e),
                "ollama": "healthy" if ollama_ok else "unreachable"
            }
        )

    all_ready = ollama_ok and chat_model_ready and embed_model_ready and postgres_ok
    status_str = "healthy" if all_ready else "degraded"
    status_code = 200 if (postgres_ok and ollama_ok) else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status_str,
            "database_engine": engine,
            "database_connected": postgres_ok,
            "ollama_host": OLLAMA_HOST,
            "ollama_online": ollama_ok,
            "models": {
                "chat_model": CHAT_MODEL,
                "embed_model": EMBED_MODEL,
                "chat_model_ready": chat_model_ready,
                "embed_model_ready": embed_model_ready,
                "available_models": models_available
            },
            "chunks_count": chunks_count,
            "invoices_count": invoices_count
        }
    )

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Primary chat endpoint with conversational history rewriting.
    Executes the Config E dual-path pipeline via chat().
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = req.session_id or "default"
    history = get_session_history(session_id)
    rewritten_q = rewrite_followup_query(req.question, history)

    t0 = time.time()
    try:
        resp = chat(
            question=rewritten_q,
            model=CHAT_MODEL,
            embed_model=EMBED_MODEL,
            retrieval_strategy=RETRIEVAL_STRATEGY,
            gate_mode=GATE_MODE,
            planner=PLANNER_MODE,
            backend=DB_BACKEND
        )
        latency = round((time.time() - t0) * 1000, 2)
        resp.latency_ms = latency

        add_to_session(session_id, "user", req.question)
        add_to_session(session_id, "assistant", resp.answer)

        logger.info(
            f"Question: '{req.question}' | Route: {resp.route} | "
            f"Latency: {latency}ms | Sources: {len(resp.sources)}"
        )
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat processing error for '{req.question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    """
    Streaming chat endpoint yielding Server-Sent Events (SSE).
    Emits status events (planning, retrieving, generating), then streams the verified answer
    in small chunks, followed by a final event containing sources.
    The deterministic guards require the complete synthesized text to verify claims;
    hence, the verified final answer is streamed to preserve correctness guarantees.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = req.session_id or "default"
    history = get_session_history(session_id)
    rewritten_q = rewrite_followup_query(req.question, history)

    def event_stream():
        yield f"event: status\ndata: {json.dumps({'status': 'planning'})}\n\n"
        time.sleep(0.01)

        yield f"event: status\ndata: {json.dumps({'status': 'retrieving'})}\n\n"
        time.sleep(0.01)

        yield f"event: status\ndata: {json.dumps({'status': 'generating'})}\n\n"

        t0 = time.time()
        resp = chat(
            question=rewritten_q,
            model=CHAT_MODEL,
            embed_model=EMBED_MODEL,
            retrieval_strategy=RETRIEVAL_STRATEGY,
            gate_mode=GATE_MODE,
            planner=PLANNER_MODE,
            backend=DB_BACKEND
        )
        latency = round((time.time() - t0) * 1000, 2)
        resp.latency_ms = latency

        # Stream verified answer in small token/word chunks
        words = resp.answer.split(" ")
        for i, word in enumerate(words):
            chunk_text = word + (" " if i < len(words) - 1 else "")
            yield f"event: chunk\ndata: {json.dumps({'text': chunk_text})}\n\n"
            time.sleep(0.01)

        # Final event with sources and metadata
        yield f"event: final\ndata: {json.dumps({'sources': resp.sources, 'route': resp.route, 'latency_ms': latency})}\n\n"

        add_to_session(session_id, "user", req.question)
        add_to_session(session_id, "assistant", resp.answer)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/search", response_model=SearchResponse)
def raw_search_endpoint(req: SearchRequest):
    """
    Raw vector/hybrid retrieval endpoint without LLM synthesis for evaluation and inspection.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    k = req.top_k or req.limit or 5
    t0 = time.time()
    try:
        raw_hits = search_chunks(
            query=req.query,
            doc_type=req.doc_type,
            limit=k,
            retrieval_mode=RETRIEVAL_STRATEGY,
            backend=DB_BACKEND
        )
        latency = round((time.time() - t0) * 1000, 2)
        logger.info(f"Raw Search: '{req.query}' | Hits: {len(raw_hits)} | Latency: {latency:.2f}ms")

        hits = [
            SearchHit(
                content=h["content"],
                source_file=h["source_file"],
                page=h["page"],
                section=h["section"],
                similarity=float(h.get("similarity", 0.0))
            )
            for h in raw_hits
        ]
        return SearchResponse(query=req.query, results=hits, count=len(hits))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search retrieval error for '{req.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF or CSV document.
    Validates file extension and size.
    For PDF: extracts pages, chunks text, generates embeddings, and inserts into chunks table.
    For CSV: parses invoice rows and inserts/upserts into invoices table.
    """
    filename = file.filename or ""
    fn_lower = filename.lower()
    if not (fn_lower.endswith(".pdf") or fn_lower.endswith(".csv")):
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds maximum 10MB limit.")

    try:
        if fn_lower.endswith(".pdf"):
            dest_path = os.path.join(POLICIES_DIR, filename)
            with open(dest_path, "wb") as f_out:
                f_out.write(content)

            pages = extract_pages(dest_path)
            new_chunks = []
            for page_num, raw_text in pages:
                cleaned = clean_text(raw_text)
                chunks = chunk_page_text(
                    source_file=filename,
                    page=page_num,
                    cleaned_text=cleaned,
                    target_chunk_tokens=500,
                    overlap_tokens=50
                )
                new_chunks.extend(chunks)

            if new_chunks:
                texts = [c["content"] for c in new_chunks]
                embs = get_embeddings(texts, model=EMBED_MODEL)

                conn, engine = get_connection(backend=DB_BACKEND)
                cur = conn.cursor()
                spec = get_embedder_spec(EMBED_MODEL)
                tbl = spec.get("table", "chunks_bge")

                if engine == "postgres":
                    insert_sql = f"""
                        INSERT INTO {tbl} (source_file, page, section, doc_type, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s);
                    """
                    data = [
                        (c["source_file"], c["page"], c["section"], c["doc_type"], c["content"], embs[i])
                        for i, c in enumerate(new_chunks)
                    ]
                    cur.executemany(insert_sql, data)
                else:
                    insert_sql = f"""
                        INSERT INTO {tbl} (source_file, page, section, doc_type, content, embedding)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """
                    data = [
                        (c["source_file"], c["page"], c["section"], c["doc_type"], c["content"], json.dumps(embs[i]))
                        for i, c in enumerate(new_chunks)
                    ]
                    cur.executemany(insert_sql, data)

                conn.commit()
                conn.close()

            logger.info(f"Ingested PDF {filename}: {len(new_chunks)} chunks created in {tbl}.")
            return UploadResponse(
                filename=filename,
                chunks_created=len(new_chunks),
                rows_inserted=0,
                status="success",
                message=f"Successfully indexed {len(new_chunks)} chunks from {filename}."
            )

        else:  # CSV upload
            import io, csv
            dest_path = os.path.join(DATA_DIR, filename)
            with open(dest_path, "wb") as f_out:
                f_out.write(content)

            text_stream = io.StringIO(content.decode("utf-8"))
            reader = csv.DictReader(text_stream)
            required_cols = {"invoice_id", "vendor_name", "amount_inr", "status"}
            if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV missing required columns: {required_cols - set(reader.fieldnames or [])}"
                )

            rows = list(reader)
            if not rows:
                raise HTTPException(status_code=400, detail="CSV file contains no data rows.")

            conn, engine = get_connection(backend=DB_BACKEND)
            cur = conn.cursor()
            if engine == "postgres":
                insert_sql = """
                    INSERT INTO invoices (invoice_id, vendor_name, invoice_date, due_date, amount_inr, status, payment_date, department, description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (invoice_id) DO UPDATE SET
                        vendor_name = EXCLUDED.vendor_name,
                        amount_inr = EXCLUDED.amount_inr,
                        status = EXCLUDED.status,
                        due_date = EXCLUDED.due_date,
                        payment_date = EXCLUDED.payment_date;
                """
                data = [
                    (
                        r["invoice_id"],
                        r["vendor_name"],
                        r.get("invoice_date"),
                        r.get("due_date"),
                        float(r["amount_inr"]),
                        r["status"],
                        r.get("payment_date") or None,
                        r.get("department"),
                        r.get("description")
                    )
                    for r in rows
                ]
                cur.executemany(insert_sql, data)
            else:
                insert_sql = """
                    INSERT OR REPLACE INTO invoices (invoice_id, vendor_name, invoice_date, due_date, amount_inr, status, payment_date, department, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """
                data = [
                    (
                        r["invoice_id"],
                        r["vendor_name"],
                        r.get("invoice_date"),
                        r.get("due_date"),
                        float(r["amount_inr"]),
                        r["status"],
                        r.get("payment_date") or None,
                        r.get("department"),
                        r.get("description")
                    )
                    for r in rows
                ]
                cur.executemany(insert_sql, data)

            conn.commit()
            conn.close()

            logger.info(f"Ingested CSV {filename}: {len(rows)} invoice rows loaded.")
            return UploadResponse(
                filename=filename,
                chunks_created=0,
                rows_inserted=len(rows),
                status="success",
                message=f"Successfully loaded {len(rows)} invoices from {filename}."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document upload and processing failed: {str(e)}")

# Legacy compatibility endpoints
@app.post("/api/query", response_model=QueryResponse)
def handle_query_legacy(req: QueryRequest):
    resp = chat(req.question)
    return QueryResponse(question=req.question, route=resp.route or "POLICY_RAG", answer=resp.answer)

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
    res = query_invoices(backend=DB_BACKEND)
    return {
        "total_count": res.get("count", 0),
        "total_amount": res.get("total_amount", 0.0),
        "status_breakdown": res.get("status_breakdown", {})
    }

# -------------------------------------------------------------
# Interactive UI
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """<!DOCTYPE html>
<html>
<head><title>Finance RAG API</title></head>
<body style="font-family: sans-serif; padding: 40px; background: #0F172A; color: #F8FAFC;">
  <h2>Finance RAG & Invoice Assistant API</h2>
  <p>The API is operational. Visit <a href="/docs" style="color: #38BDF8;">/docs</a> for interactive OpenAPI documentation.</p>
</body>
</html>"""
