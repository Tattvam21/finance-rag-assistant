import os
import sys
import sqlite3
import logging
from typing import Optional, Tuple, Any
from app.config import DATABASE_URL, EMBED_DIM

logger = logging.getLogger("finance_rag.db")

SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "finance_rag.db")

def is_postgres_available() -> bool:
    try:
        import psycopg
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            return True
    except Exception:
        return False

def get_connection(backend: Optional[str] = None) -> Tuple[Any, str]:
    """
    Returns a Postgres or SQLite database connection.
    Postgres is the default production backend with pgvector.
    SQLite is only permitted when explicitly requested via backend='sqlite'
    or environment variable DB_BACKEND='sqlite'.
    """
    selected_backend = (backend or os.getenv("DB_BACKEND", "postgres")).strip().lower()

    if selected_backend == "sqlite":
        print("[WARNING] Running with local SQLite backend (--backend sqlite). In-memory vector calculation active.")
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn, "sqlite"

    # Default: Postgres (must be reachable)
    if not is_postgres_available():
        sep = "=" * 70
        sys.stderr.write(
            f"\n{sep}\n"
            "[FATAL ERROR] PostgreSQL with pgvector is unreachable at:\n"
            f"    {DATABASE_URL}\n\n"
            "Silent fallback to SQLite is disabled.\n"
            "To launch the PostgreSQL database container with pgvector, run:\n"
            "    docker compose up -d db\n\n"
            "Alternatively, run explicitly with SQLite using:\n"
            "    --backend sqlite\n"
            f"{sep}\n\n"
        )
        sys.exit(1)

    import psycopg
    from pgvector.psycopg import register_vector
    conn = psycopg.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
    except Exception:
        pass
    register_vector(conn)
    return conn, "postgres"

def init_db(backend: Optional[str] = None) -> str:
    """Initializes tables, vector indexes, and full-text search indexes in the active engine."""
    conn, engine = get_connection(backend=backend)
    cur = conn.cursor()

    if engine == "postgres":
        cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id VARCHAR(50) PRIMARY KEY,
                vendor_name VARCHAR(255) NOT NULL,
                invoice_date DATE NOT NULL,
                due_date DATE NOT NULL,
                amount_inr NUMERIC(12, 2) NOT NULL,
                status VARCHAR(50) NOT NULL,
                payment_date DATE,
                department VARCHAR(100),
                description TEXT
            );
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                source_file TEXT,
                page INT,
                section TEXT,
                doc_type TEXT,
                content TEXT,
                embedding vector({EMBED_DIM})
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw 
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """)
        # Postgres full-text search index
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_fts_idx 
            ON chunks USING gin(to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')));
        """)
        conn.commit()
    else:
        cur.execute("DROP TABLE IF EXISTS document_chunks;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id TEXT PRIMARY KEY,
                vendor_name TEXT NOT NULL,
                invoice_date TEXT NOT NULL,
                due_date TEXT NOT NULL,
                amount_inr REAL NOT NULL,
                status TEXT NOT NULL,
                payment_date TEXT,
                department TEXT,
                description TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                page INTEGER,
                section TEXT,
                doc_type TEXT,
                content TEXT,
                embedding TEXT
            );
        """)
        # SQLite FTS5 table
        try:
            cur.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    section,
                    source_file,
                    content='chunks',
                    content_rowid='id'
                );
            """)
        except Exception:
            pass
        conn.commit()

    conn.close()
    return engine
