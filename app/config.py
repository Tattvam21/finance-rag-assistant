import os
import sys
import logging
from typing import Tuple, Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("finance_rag.config")

# Ollama Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
GATE_MODEL = os.getenv("GATE_MODEL", CHAT_MODEL)
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
GATE_MODE = os.getenv("GATE_MODE", "merged")
PLANNER_MODE = os.getenv("PLANNER_MODE", "auto")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))

# Policy & Tool Settings
LATE_FEE_MONTHLY_RATE = float(os.getenv("LATE_FEE_MONTHLY_RATE", "0.02"))  # 2.0% per month / 24% annual
LATE_FEE_GRACE_PERIOD_DAYS = int(os.getenv("LATE_FEE_GRACE_PERIOD_DAYS", "0"))
AS_OF_DATE = os.getenv("AS_OF_DATE", "2024-08-05")

# Database Settings
DB_BACKEND = os.getenv("DB_BACKEND", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "finance_rag")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

_ollama_client = None

def get_ollama_client():
    """Returns a singleton Ollama client configured with OLLAMA_HOST and timeout."""
    global _ollama_client
    if _ollama_client is None:
        import ollama
        _ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    return _ollama_client

def get_ollama_running_info() -> List[Dict[str, Any]]:
    """Retrieves runtime model statistics similar to `ollama ps` via /api/ps."""
    import httpx
    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/ps", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("models", [])
    except Exception:
        pass
    return []

def run_startup_checks(
    backend: Optional[str] = None,
    chat_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    embed_dim: Optional[int] = None,
    retrieval_mode: Optional[str] = None,
    gate_mode: Optional[str] = None,
    planner_mode: Optional[str] = None,
    threshold: Optional[float] = None,
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validates startup requirements:
    1. Confirms Ollama server reachability and model availability.
    2. Prints active running model status and memory footprint.
    3. Confirms database backend status.
    """
    selected_backend = (backend or DB_BACKEND).lower()
    cur_chat_model = chat_model or CHAT_MODEL
    cur_embed_model = embed_model or EMBED_MODEL
    cur_embed_dim = embed_dim or EMBED_DIM
    cur_retrieval_mode = retrieval_mode or RETRIEVAL_STRATEGY
    cur_gate_mode = gate_mode or GATE_MODE
    cur_planner_mode = planner_mode or PLANNER_MODE
    cur_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    cur_as_of_date = as_of_date or AS_OF_DATE

    status_report = {
        "ollama_reachable": False,
        "chat_model_available": False,
        "embed_model_available": False,
        "missing_models": [],
        "database_backend": selected_backend,
        "database_online": False,
        "running_models": []
    }

    print("------------------------------------------------------------")
    print("Running System Startup Checks...")
    print(f"  Ollama Host:    {OLLAMA_HOST}")
    print(f"  Chat Model:     {cur_chat_model}")
    print(f"  Embed Model:    {cur_embed_model} (Dim: {cur_embed_dim})")
    print(f"  Retrieval Mode: {cur_retrieval_mode}")
    print(f"  Gate Mode:      {cur_gate_mode}")
    print(f"  Planner Mode:   {cur_planner_mode}")
    print(f"  Similarity Cut: {cur_threshold}")
    print(f"  AS_OF_DATE:     {cur_as_of_date}")
    print(f"  Vector Backend: {selected_backend.upper()}")

    # 1. Check Ollama Reachability and Models
    try:
        client = get_ollama_client()
        models_resp = client.list()
        status_report["ollama_reachable"] = True
        print("  [✓] Ollama server is reachable")

        installed_model_names = [m.model for m in models_resp.models]
        def is_installed(target: str) -> bool:
            target_clean = target.lower()
            return any(
                target_clean == name.lower() or name.lower().startswith(f"{target_clean}:") or target_clean.startswith(f"{name.lower()}:")
                for name in installed_model_names
            )

        if is_installed(CHAT_MODEL):
            status_report["chat_model_available"] = True
            print(f"  [✓] Chat model '{CHAT_MODEL}' is ready")
        else:
            status_report["missing_models"].append(CHAT_MODEL)
            print(f"  [✗] Chat model '{CHAT_MODEL}' is NOT pulled.")

        if is_installed(EMBED_MODEL):
            status_report["embed_model_available"] = True
            print(f"  [✓] Embedding model '{EMBED_MODEL}' is ready")
        else:
            status_report["missing_models"].append(EMBED_MODEL)
            print(f"  [✗] Embedding model '{EMBED_MODEL}' is NOT pulled.")

        # Show active running models
        running = get_ollama_running_info()
        status_report["running_models"] = running
        if running:
            for rm in running:
                name = rm.get("name", "unknown")
                size_mb = round(rm.get("size", 0) / (1024 * 1024), 1)
                proc = rm.get("details", {}).get("quantization_level", "default")
                print(f"  [i] Active Model in Memory: {name} ({size_mb} MB, {proc})")

    except Exception as e:
        status_report["ollama_reachable"] = False
        print(f"  [✗] Cannot connect to Ollama at {OLLAMA_HOST}: {e}")

    # 2. Check Database Connectivity
    if selected_backend == "postgres":
        from app.db import is_postgres_available
        if is_postgres_available():
            status_report["database_online"] = True
            print("  [✓] PostgreSQL (pgvector) connected successfully")
        else:
            print("  [✗] PostgreSQL is OFFLINE.")
            print("      To launch: docker compose up -d db")
            print("      Or use SQLite fallback: --backend sqlite")
    else:
        status_report["database_online"] = True
        print("  [i] SQLite backend in use (--backend sqlite)")

    print("------------------------------------------------------------\n")
    return status_report
