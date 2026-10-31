import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import EMBED_DIM
from app.db import get_connection
from scripts.ingest_data import load_invoices_table
from app.ingestion.reindex import reindex_all_chunks

def reset_database():
    print("=" * 65)
    print("RESETTING DATABASE SCHEMA AND RE-INDEXING EMBEDDINGS")
    print(f"Target Vector Dimension: {EMBED_DIM}")
    print("=" * 65)

    conn, engine = get_connection()
    cur = conn.cursor()

    if engine == "postgres":
        print("Dropping and recreating Postgres tables...")
        cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
        cur.execute("DROP TABLE IF EXISTS invoices CASCADE;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE chunks (
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
            CREATE INDEX chunks_embedding_hnsw 
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """)
        cur.execute("""
            CREATE TABLE invoices (
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
        conn.commit()
    else:
        print("Resetting local SQLite tables...")
        cur.execute("DROP TABLE IF EXISTS chunks;")
        cur.execute("DROP TABLE IF EXISTS invoices;")
        cur.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                page INTEGER,
                section TEXT,
                doc_type TEXT,
                content TEXT,
                embedding TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE invoices (
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
        conn.commit()

    conn.close()
    print("Tables reset successfully.")

    # Reload data
    load_invoices_table()
    reindex_all_chunks()

    print("=" * 65)
    print("DATABASE RESET & INGESTION COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    reset_database()
