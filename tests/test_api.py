import io
import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.llm.schemas import ChatResponse, InvoiceRow

client = TestClient(app)

def test_health_endpoint():
    """GET /health should return 200 with database, ollama, and required models info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "database_engine" in data
    assert "models" in data
    assert "chat_model" in data["models"]
    assert "embed_model" in data["models"]
    assert "chunks_count" in data
    assert "invoices_count" in data

def test_search_endpoint_valid():
    """POST /search with valid query should return retrieval hits without LLM calls."""
    response = client.post("/search", json={"query": "payment terms", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "payment terms"
    assert "results" in data
    assert len(data["results"]) <= 3
    assert "count" in data
    if data["results"]:
        hit = data["results"][0]
        assert "content" in hit
        assert "source_file" in hit
        assert "similarity" in hit

def test_search_endpoint_empty_query():
    """POST /search with empty query should return HTTP 400."""
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data

def test_chat_endpoint_mocked():
    """POST /chat with LLM mocked should return answer, sources, route, and latency_ms."""
    mock_resp = ChatResponse(
        answer="Standard payment terms are Net 30 days.",
        invoices=[],
        sources=["payment_policy.pdf, p.1, §1.1"],
        route="PLANNER (planner)",
        latency_ms=120.5
    )

    with patch("app.api.main.chat", return_value=mock_resp):
        response = client.post("/chat", json={"question": "What is the payment period?", "session_id": "test_sess"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Standard payment terms are Net 30 days."
        assert data["sources"] == ["payment_policy.pdf, p.1, §1.1"]
        assert data["route"] == "PLANNER (planner)"
        assert "latency_ms" in data
        assert data["latency_ms"] > 0

def test_chat_endpoint_empty_question():
    """POST /chat with empty question should return HTTP 400."""
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 400
    assert "detail" in response.json()

def test_upload_validation_unsupported_type():
    """POST /documents/upload with non-PDF and non-CSV file should return HTTP 400."""
    fake_file = io.BytesIO(b"Hello world")
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", fake_file, "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF and CSV files are supported" in response.json()["detail"]

def test_upload_validation_empty_file():
    """POST /documents/upload with 0-byte file should return HTTP 400."""
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", empty_file, "application/pdf")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_upload_validation_file_too_large():
    """POST /documents/upload with file exceeding 10MB should return HTTP 413."""
    large_file = io.BytesIO(b"0" * (10 * 1024 * 1024 + 100))
    response = client.post(
        "/documents/upload",
        files={"file": ("large.pdf", large_file, "application/pdf")}
    )
    assert response.status_code == 413
    assert "10MB" in response.json()["detail"]

def test_upload_csv_valid():
    """POST /documents/upload with valid invoice CSV should insert rows and return count."""
    csv_content = (
        "invoice_id,vendor_name,invoice_date,due_date,amount_inr,status,payment_date,department,description\n"
        "INV-TEST-999,Test Vendor Inc,2024-01-01,2024-02-01,15000.00,Pending,,Engineering,Testing upload\n"
    )
    file_obj = io.BytesIO(csv_content.encode("utf-8"))
    try:
        response = client.post(
            "/documents/upload",
            files={"file": ("test_invoices.csv", file_obj, "text/csv")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["rows_inserted"] == 1
        assert data["chunks_created"] == 0
    finally:
        # Clean up inserted test record so database row count stays pristine
        import os
        from app.db import get_connection
        conn, _ = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM invoices WHERE invoice_id = 'INV-TEST-999';")
        conn.commit()
        conn.close()
        test_csv_path = os.path.join("data", "test_invoices.csv")
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)

def test_chat_stream_event_order():
    """
    POST /chat/stream should emit status events (planning, retrieving, generating),
    followed by chunk events, and conclude with a final event containing sources.
    """
    mock_resp = ChatResponse(
        answer="Settlement is scheduled thirty days from invoice receipt.",
        sources=["payment_policy.pdf, p.1, §1.1"],
        route="PLANNER (planner)",
        latency_ms=95.0
    )

    with patch("app.api.main.chat", return_value=mock_resp):
        response = client.post("/chat/stream", json={"question": "What is the payment period?"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body = response.text
        blocks = [b.strip() for b in body.split("\n\n") if b.strip()]

        events = []
        for block in blocks:
            lines = block.split("\n")
            event_type = None
            data_val = None
            for line in lines:
                if line.startswith("event:"):
                    event_type = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    data_val = line.replace("data:", "").strip()
            events.append((event_type, data_val))

        # Check event sequence
        status_events = [json.loads(data)["status"] for ev, data in events if ev == "status"]
        assert status_events == ["planning", "retrieving", "generating"]

        chunk_events = [json.loads(data)["text"] for ev, data in events if ev == "chunk"]
        full_text = "".join(chunk_events).strip()
        assert full_text == "Settlement is scheduled thirty days from invoice receipt."

        final_events = [json.loads(data) for ev, data in events if ev == "final"]
        assert len(final_events) == 1
        assert final_events[0]["sources"] == ["payment_policy.pdf, p.1, §1.1"]
        assert final_events[0]["route"] == "PLANNER (planner)"
