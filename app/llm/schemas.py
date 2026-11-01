from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class InvoiceRow(BaseModel):
    invoice_id: str
    vendor_name: str
    invoice_date: Any
    due_date: Any
    amount_inr: float
    status: str
    payment_date: Optional[Any] = None
    department: Optional[str] = None
    description: Optional[str] = None

class InvoiceAnswer(BaseModel):
    answer: str
    invoices: List[InvoiceRow] = []
    sources: List[str] = []

class ChatRequest(BaseModel):
    question: str = Field(..., description="User query about finance policies or invoices")
    session_id: Optional[str] = Field("default", description="Session identifier for multi-turn conversational history")

class ChatResponse(BaseModel):
    answer: str
    invoices: List[InvoiceRow] = []
    sources: List[str] = []
    route: Optional[str] = None
    latency_ms: Optional[float] = None
    retrieved_chunks: List[Dict[str, Any]] = []
    queries_run: List[str] = []
    plan: Optional[Dict[str, Any]] = None
    plan_source: Optional[str] = None
    tool_results: List[Any] = []
    ollama_calls: int = 0
    stage_timings: Dict[str, float] = {}
    prompt_tokens: int = 0

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    limit: Optional[int] = 5
    doc_type: Optional[str] = None

class SearchHit(BaseModel):
    content: str
    source_file: str
    page: int
    section: str
    similarity: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]
    count: int

class UploadResponse(BaseModel):
    filename: str
    chunks_created: int = 0
    rows_inserted: int = 0
    status: str
    message: Optional[str] = None

# Legacy schemas for backward compatibility
class QueryRequest(BaseModel):
    question: str = Field(..., description="User's finance or invoice question")

class Citation(BaseModel):
    doc_name: str
    section_num: str
    section_title: str

class QueryResponse(BaseModel):
    question: str
    route: str = Field(..., description="Routing path: POLICY_RAG or INVOICE_DATA")
    answer: str
    citations: List[Citation] = []
    data: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None

# Query Planner and Structured Generation Schemas
class InvoiceFilter(BaseModel):
    invoice_id: Optional[str] = None
    vendor: Optional[str] = None
    status: Optional[List[str]] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    due_before: Optional[str] = None

class QueryPlan(BaseModel):
    needs_policy: bool = False
    policy_query: str = ""
    invoice: Optional[InvoiceFilter] = None
    late_fee_invoice_id: Optional[str] = None

class StructuredAnswer(BaseModel):
    answerable: bool = True
    answer: str = ""
    supporting_quote: str = ""
    cited_chunk_ids: List[int] = []
