import os
import sys
import re
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import EMBED_DIM, EMBED_MODEL, DB_BACKEND, get_ollama_client
from app.db import get_connection, init_db
from app.ingestion.parse import extract_pages, clean_text
from app.ingestion.chunk import chunk_page_text

logger = logging.getLogger("finance_rag.reindex")
logging.basicConfig(level=logging.INFO)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
POLICIES_DIR = os.path.join(DATA_DIR, "policies")

EMBEDDER_CONFIGS = {
    "nomic-embed-text": {
        "table": "chunks",
        "dim": 768,
        "doc_prefix": "search_document: ",
    },
    "mxbai-embed-large": {
        "table": "chunks_mxbai",
        "dim": 1024,
        "doc_prefix": "",
    },
    "bge-m3": {
        "table": "chunks_bge",
        "dim": 1024,
        "doc_prefix": "",
    },
}

SAMPLE_QUESTIONS = [
    "What is the payment period?",
    "What is the late fee policy and when is it charged?",
    "What are the travel and per diem allowance rates?",
    "What is the TDS withholding tax rate?",
    "What is the invoice dispute resolution procedure?",
    "What is the standard GST rate in India?",
    "What is the CEO's salary and personal home address?"
]

def ensure_embedder_table(engine, conn, cur, table_name: str, dim: int):
    """Creates dedicated table and vector/FTS indexes for an embedder."""
    if engine == "postgres":
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                source_file TEXT,
                page INT,
                section TEXT,
                doc_type TEXT,
                content TEXT,
                embedding vector({dim})
            );
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw 
            ON {table_name} USING hnsw (embedding vector_cosine_ops);
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_fts_idx 
            ON {table_name} USING gin(to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')));
        """)
    else:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT,
                page INTEGER,
                section TEXT,
                doc_type TEXT,
                content TEXT,
                embedding TEXT
            );
        """)
        try:
            cur.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_fts USING fts5(
                    content,
                    section,
                    source_file,
                    content='{table_name}',
                    content_rowid='id'
                );
            """)
        except Exception:
            pass
    conn.commit()

def generate_embeddings_for_model(model_name: str, texts: List[str], doc_prefix: str, batch_size: int = 16) -> List[List[float]]:
    """Generates embeddings using the specified Ollama model and document prefix."""
    client = get_ollama_client()
    all_embeddings: List[List[float]] = []
    prefixed_texts = [f"{doc_prefix}{t.strip()}" for t in texts]

    for i in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[i : i + batch_size]
        try:
            resp = client.embed(model=model_name, input=batch)
            if not resp.embeddings or len(resp.embeddings) != len(batch):
                raise RuntimeError(
                    f"Batch embed returned {len(resp.embeddings) if resp.embeddings else 0} embeddings "
                    f"for batch size {len(batch)}"
                )
            all_embeddings.extend(resp.embeddings)
        except Exception:
            # Fallback to single embedding endpoint if embed() fails
            batch_embs = []
            for item in batch:
                r = client.embeddings(model=model_name, prompt=item)
                batch_embs.append(r["embedding"])
            all_embeddings.extend(batch_embs)

    return all_embeddings

def extract_and_validate_chunks() -> List[Dict[str, Any]]:
    """
    Parses and chunks all PDF policy documents from POLICIES_DIR.
    Enforces strict ingestion assertions:
      1. Fails if any chunk matches (?<![A-Za-z])I\\d
      2. Fails if any chunk contains '■'
      3. Fails if any chunk contains fewer than 60 characters or lacks sentence structure
    """
    if not os.path.exists(POLICIES_DIR):
        raise FileNotFoundError(f"Directory {POLICIES_DIR} does not exist.")

    pdf_files = [f for f in sorted(os.listdir(POLICIES_DIR)) if f.endswith(".pdf")]
    all_chunks: List[Dict[str, Any]] = []

    global_chunk_idx = 0
    raw_pdf_texts: Dict[str, str] = {}
    for pdf_name in pdf_files:
        pdf_path = os.path.join(POLICIES_DIR, pdf_name)
        pages = extract_pages(pdf_path)
        raw_pdf_texts[pdf_name] = "\n".join([t for _, t in pages])

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

    # Ingestion Assertions
    for c in all_chunks:
        content = c["content"]
        # Assertion 1: no corrupted currency symbol like I5,000
        if re.search(r'(?<![A-Za-z])I\d', content):
            raise AssertionError(f"Ingestion assertion failed: chunk matches (?<![A-Za-z])I\\d in {c['source_file']} §{c['section']}:\n{content}")
        # Assertion 2: no corrupted rupee glyph ■
        if "■" in content:
            raise AssertionError(f"Ingestion assertion failed: chunk contains corrupted glyph '■' in {c['source_file']} §{c['section']}:\n{content}")
        # Assertion 3: minimum 60 characters
        if len(content.strip()) < 60:
            raise AssertionError(f"Ingestion assertion failed: chunk is shorter than 60 chars ({len(content.strip())} chars) in {c['source_file']} §{c['section']}:\n{content}")
        # Assertion 4: at least one sentence
        if not re.search(r'[.!?:]', content.strip()):
            raise AssertionError(f"Ingestion assertion failed: chunk lacks sentence terminator in {c['source_file']} §{c['section']}:\n{content}")
        # Assertion 5 (Strict): every number token in every chunk must appear in the SAME section of the raw PDF text
        src_pdf_text = raw_pdf_texts[c["source_file"]]
        sec_m = re.match(r"^(\d+\.\d+)", c["section"].strip())
        if sec_m:
            sec_num = sec_m.group(1)
            start_m = re.search(r"(?:^|\n)\s*" + re.escape(sec_num) + r"\b", src_pdf_text)
            if start_m:
                next_m = re.search(r"(?:^|\n)\s*\d+\.\d+\b", src_pdf_text[start_m.end():])
                sec_raw_text = src_pdf_text[start_m.start() : start_m.end() + next_m.start()] if next_m else src_pdf_text[start_m.start():]
            else:
                sec_raw_text = src_pdf_text
        else:
            sec_raw_text = src_pdf_text

        for num_token in re.findall(r'\b\d[\d,]*(?:\.\d+)?\b', content):
            norm_num = num_token.replace(',', '')
            if num_token not in sec_raw_text and norm_num not in sec_raw_text:
                raise AssertionError(
                    f"Ingestion assertion failed: number token '{num_token}' in {c['source_file']} §{c['section']} "
                    f"does not appear in the same section of the raw PDF text."
                )

    # Print chunk counts per file
    chunk_counts_by_file: Dict[str, int] = {}
    for c in all_chunks:
        f = c["source_file"]
        chunk_counts_by_file[f] = chunk_counts_by_file.get(f, 0) + 1

    print("------------------------------------------------------------")
    print(f"Extracted and validated {len(all_chunks)} chunks across {len(pdf_files)} PDF policies:")
    for f, count in sorted(chunk_counts_by_file.items()):
        print(f"  • {f}: {count} chunks")
    print("------------------------------------------------------------")

    return all_chunks

def reindex_backend_tables(backend: str, all_chunks: List[Dict[str, Any]], embedder_names: List[str], batch_size: int = 16):
    """Re-indexes specified embedder tables into the given backend."""
    init_db(backend=backend)
    conn, engine = get_connection(backend=backend)
    cur = conn.cursor()

    # Drop legacy document_chunks table
    if engine == "postgres":
        cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
    else:
        cur.execute("DROP TABLE IF EXISTS document_chunks;")
    conn.commit()

    texts = [c["content"] for c in all_chunks]

    for model_name in embedder_names:
        cfg = EMBEDDER_CONFIGS[model_name]
        table = cfg["table"]
        dim = cfg["dim"]
        doc_prefix = cfg["doc_prefix"]

        ensure_embedder_table(engine, conn, cur, table, dim)

        print(f"[{engine.upper()}] Re-indexing {table} with '{model_name}' ({dim} dims)...")
        cur.execute(f"DELETE FROM {table};")
        if engine == "sqlite":
            try:
                cur.execute(f"DELETE FROM {table}_fts;")
            except Exception:
                pass
        conn.commit()

        embeddings = generate_embeddings_for_model(model_name, texts, doc_prefix=doc_prefix, batch_size=batch_size)

        if engine == "postgres":
            insert_sql = f"""
                INSERT INTO {table} (id, source_file, page, section, doc_type, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            data = [
                (i + 1, c["source_file"], c["page"], c["section"], c["doc_type"], c["content"], embeddings[i])
                for i, c in enumerate(all_chunks)
            ]
            cur.executemany(insert_sql, data)
        else:
            insert_sql = f"""
                INSERT INTO {table} (id, source_file, page, section, doc_type, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """
            data = [
                (i + 1, c["source_file"], c["page"], c["section"], c["doc_type"], c["content"], json.dumps(embeddings[i]))
                for i, c in enumerate(all_chunks)
            ]
            cur.executemany(insert_sql, data)

            try:
                cur.execute(f"INSERT INTO {table}_fts({table}_fts) VALUES('rebuild');")
            except Exception:
                pass

        conn.commit()
        print(f"[{engine.upper()}] Successfully indexed {len(all_chunks)} chunks in '{table}'.")

    conn.close()

def reindex_all(backend: str = None, all_embedders: bool = True, batch_size: int = 16) -> List[Dict[str, Any]]:
    """
    Main reindex routine:
    1. Validates current PDFs and chunks with strict assertions.
    2. Drops legacy document_chunks table.
    3. Re-indexes Postgres AND SQLite for every embedder table (nomic, mxbai, bge-m3).
    """
    all_chunks = extract_and_validate_chunks()
    models = list(EMBEDDER_CONFIGS.keys()) if all_embedders else [EMBED_MODEL]

    backends_to_index = [backend] if backend else ["postgres", "sqlite"]
    for b in backends_to_index:
        try:
            reindex_backend_tables(backend=b, all_chunks=all_chunks, embedder_names=models, batch_size=batch_size)
        except Exception as e:
            print(f"[WARNING] Reindex failed for backend '{b}': {e}")

    return all_chunks

def evaluate_sample_questions(backend: str = None):
    """
    Evaluates sample questions against newly re-indexed chunks to report
    top-1 scores and calibrate SIMILARITY_THRESHOLD.
    """
    from app.retrieval.search import search_chunks_with_timing

    selected_backend = backend or os.getenv("DB_BACKEND", "postgres")
    print("=" * 75)
    print(f"TOP-1 SCORE BENCHMARK ON RE-INDEXED CORPUS ({selected_backend.upper()})")
    print("=" * 75)

    scores = []
    for q in SAMPLE_QUESTIONS:
        chunks, emb_ms, ret_ms = search_chunks_with_timing(q, limit=1, use_reranker=False, backend=selected_backend)
        if chunks:
            top = chunks[0]
            score = top["similarity"]
            scores.append((q, score, top["source_file"], top["section"]))
            print(f"Q: '{q}'")
            print(f"   Top-1: {score:.4f} | {top['source_file']} (§{top['section']})")
            print(f"   Latency: Embed {emb_ms}ms + Retrieval {ret_ms}ms")
        else:
            print(f"Q: '{q}' -> No chunks found")

    print("-" * 75)
    valid_domain_scores = [s[1] for s in scores[:-1]]
    unanswerable_score = scores[-1][1] if scores else 0.0

    min_valid = min(valid_domain_scores) if valid_domain_scores else 0.0
    avg_valid = sum(valid_domain_scores) / len(valid_domain_scores) if valid_domain_scores else 0.0

    print(f"In-domain Questions Top-1 Range: {min(valid_domain_scores):.4f} to {max(valid_domain_scores):.4f} (Avg: {avg_valid:.4f})")
    print(f"Out-of-domain Control Question: {unanswerable_score:.4f}")
    print("=" * 75 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Re-index document chunks across embedding models and backends")
    parser.add_argument("--backend", type=str, default=None, help="Database backend: 'postgres', 'sqlite', or both (default)")
    parser.add_argument("--all-embedders", action="store_true", default=True, help="Re-index all 3 embedder tables: nomic, mxbai, bge-m3")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for Ollama embeddings")
    args = parser.parse_args()

    reindex_all(backend=args.backend, all_embedders=args.all_embedders, batch_size=args.batch_size)
    evaluate_sample_questions(backend=args.backend or "postgres")

if __name__ == "__main__":
    main()
