import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
from typing import List, Dict, Any
from app.db import get_connection, init_db
from app.ingestion.parse import extract_pages, clean_text
from app.ingestion.chunk import chunk_page_text
from app.ingestion.embed import get_embeddings

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "invoices.csv")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")

def load_invoices_table(backend: str = None):
    backends = [backend] if backend else ["postgres", "sqlite"]
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for b in backends:
        try:
            conn, engine = get_connection(backend=b)
            cur = conn.cursor()
            print(f"Loading {len(rows)} invoices into database ({engine})...")
            cur.execute("DELETE FROM invoices;")
            if engine == "postgres":
                insert_sql = """
                    INSERT INTO invoices (invoice_id, vendor_name, invoice_date, due_date, amount_inr, status, payment_date, department, description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (invoice_id) DO NOTHING;
                """
                data = [
                    (
                        r["invoice_id"],
                        r["vendor_name"],
                        r["invoice_date"],
                        r["due_date"],
                        float(r["amount_inr"]),
                        r["status"],
                        r["payment_date"] if r["payment_date"] else None,
                        r["department"],
                        r["description"]
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
                        r["invoice_date"],
                        r["due_date"],
                        float(r["amount_inr"]),
                        r["status"],
                        r["payment_date"] if r["payment_date"] else None,
                        r["department"],
                        r["description"]
                    )
                    for r in rows
                ]
                cur.executemany(insert_sql, data)

            conn.commit()
            conn.close()
            print(f"Invoices successfully loaded into {engine.upper()}.")
        except Exception as e:
            print(f"[WARNING] Could not load invoices into {b}: {e}")

def ingest_policy_pdfs(batch_size: int = 16):
    conn, engine = get_connection()
    cur = conn.cursor()

    # Clear existing chunks
    cur.execute("DELETE FROM chunks;")
    conn.commit()

    pdf_files = [f for f in os.listdir(POLICIES_DIR) if f.endswith(".pdf")]
    all_chunks: List[Dict[str, Any]] = []

    global_chunk_idx = 0
    for pdf_name in sorted(pdf_files):
        pdf_path = os.path.join(POLICIES_DIR, pdf_name)
        pages = extract_pages(pdf_path)
        
        for page_num, raw_text in pages:
            cleaned = clean_text(raw_text)
            chunks = chunk_page_text(
                source_file=pdf_name,
                page=page_num,
                cleaned_text=cleaned,
                target_chunk_tokens=500,
                overlap_tokens=50,
                starting_chunk_index=global_chunk_idx
            )
            all_chunks.extend(chunks)
            global_chunk_idx += len(chunks)

    print(f"Extracted {len(all_chunks)} chunks from {len(pdf_files)} PDF files.")
    print(f"Embedding in batches of {batch_size}...")

    # Batch embedding
    embeddings: List[List[float]] = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_texts = [c["content"] for c in batch]
        batch_embs = get_embeddings(batch_texts)
        embeddings.extend(batch_embs)
        print(f"  Processed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks...")

    # Insert into database
    if engine == "postgres":
        insert_sql = """
            INSERT INTO chunks (source_file, page, section, doc_type, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        data = [
            (
                c["source_file"],
                c["page"],
                c["section"],
                c["doc_type"],
                c["content"],
                embeddings[idx]
            )
            for idx, c in enumerate(all_chunks)
        ]
        cur.executemany(insert_sql, data)
    else:
        insert_sql = """
            INSERT INTO chunks (source_file, page, section, doc_type, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?);
        """
        data = [
            (
                c["source_file"],
                c["page"],
                c["section"],
                c["doc_type"],
                c["content"],
                json.dumps(embeddings[idx])
            )
            for idx, c in enumerate(all_chunks)
        ]
        cur.executemany(insert_sql, data)

    conn.commit()
    conn.close()
    print(f"Ingested {len(all_chunks)} chunks into 'chunks' table successfully.")

def main():
    engine = init_db()
    print(f"Initialized database with engine: {engine}")
    load_invoices_table()
    ingest_policy_pdfs()
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
