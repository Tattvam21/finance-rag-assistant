"""
eval/retrieval_eval.py
Retrieval benchmark comparing embedding models and retrieval modes on finance policy chunks.
Models:
  - nomic-embed-text (768 dims, search_document: / search_query:)
  - mxbai-embed-large (1024 dims, Represent this sentence for searching relevant passages: / no prefix for doc)
  - bge-m3 (1024 dims, symmetric: no prefix for doc or query)

Modes:
  - vector-only: cosine distance in pgvector
  - hybrid: vector + postgres FTS merged with RRF (k=60)

Separate tables per embedder:
  - chunks (existing active index for nomic-embed-text, 768 dims)
  - chunks_mxbai (1024 dims)
  - chunks_bge (1024 dims)
"""

import os
import sys
import re
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_connection
from app.config import get_ollama_client
from app.retrieval.search import reciprocal_rank_fusion

# Model configuration
MODEL_CONFIGS = {
    "nomic-embed-text": {
        "table": "chunks",  # Active index
        "dim": 768,
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    },
    "mxbai-embed-large": {
        "table": "chunks_mxbai",
        "dim": 1024,
        "doc_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "bge-m3": {
        "table": "chunks_bge",
        "dim": 1024,
        "doc_prefix": "",
        "query_prefix": "",
    },
}

TARGET_QUESTION_IDS = ["P07", "H03", "H08"]


def ensure_table_and_indexes(table_name: str, dim: int):
    """Creates a dedicated table and pgvector/FTS indexes for an embedder if not exists."""
    conn, _ = get_connection()
    with conn.cursor() as cur:
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
        conn.commit()


def get_embedding(model_name: str, text: str, is_query: bool = False) -> List[float]:
    """Generates an embedding with model-specific prefixes via Ollama."""
    cfg = MODEL_CONFIGS[model_name]
    prefix = cfg["query_prefix"] if is_query else cfg["doc_prefix"]
    full_text = prefix + text
    client = get_ollama_client()
    resp = client.embeddings(model=model_name, prompt=full_text)
    return resp["embedding"]


def populate_table_if_empty(model_name: str):
    """Copies chunk content from the canonical `chunks` table and generates embeddings with model_name."""
    cfg = MODEL_CONFIGS[model_name]
    table = cfg["table"]
    dim = cfg["dim"]

    ensure_table_and_indexes(table, dim)
    conn, _ = get_connection()
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table};")
        cnt = cur.fetchone()[0]
        if cnt > 0:
            return  # Already populated

        print(f"[{model_name}] Populating {table} with embeddings (dim={dim})...")
        # Fetch canonical chunks
        cur.execute("SELECT id, source_file, page, section, doc_type, content FROM chunks ORDER BY id;")
        chunks = cur.fetchall()

        for cid, src, page, sec, dtype, content in chunks:
            embed_text = f"{sec or ''}\n{content}"
            emb = get_embedding(model_name, embed_text, is_query=False)
            cur.execute(f"""
                INSERT INTO {table} (id, source_file, page, section, doc_type, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (cid, src, page, sec, dtype, content, emb))

        conn.commit()
        print(f"[{model_name}] Successfully indexed {len(chunks)} chunks in {table}.")


def retrieve_candidates(
    model_name: str,
    query: str,
    mode: str = "vector",
    limit: int = 23
) -> List[Dict[str, Any]]:
    """Runs vector or hybrid retrieval against the model's dedicated table."""
    cfg = MODEL_CONFIGS[model_name]
    table = cfg["table"]

    # Generate query embedding
    q_vec = get_embedding(model_name, query, is_query=True)

    conn, _ = get_connection()
    with conn.cursor() as cur:
        # 1. Vector candidates
        sql_vec = f"""
            SELECT source_file, page, section, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        cur.execute(sql_vec, (q_vec, q_vec, limit))
        vector_candidates = [
            {
                "source_file": r[0],
                "page": r[1],
                "section": r[2],
                "content": r[3],
                "similarity": round(float(r[4]), 4)
            }
            for r in cur.fetchall()
        ]

        if mode == "vector":
            return vector_candidates

        # 2. FTS candidates
        words = [w for w in query.split() if w.isalnum()]
        fts_candidates = []
        if words:
            sql_fts_and = f"""
                SELECT source_file, page, section, content,
                       ts_rank(to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')), plainto_tsquery('english', %s)) AS rank
                FROM {table}
                WHERE to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s;
            """
            clean_q = " ".join(words)
            cur.execute(sql_fts_and, (clean_q, clean_q, limit))
            fts_rows = cur.fetchall()
            if not fts_rows and len(words) > 1:
                or_query = " | ".join(words)
                sql_fts_or = f"""
                    SELECT source_file, page, section, content,
                           ts_rank(to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')), to_tsquery('english', %s)) AS rank
                    FROM {table}
                    WHERE to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')) @@ to_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s;
                """
                cur.execute(sql_fts_or, (or_query, or_query, limit))
                fts_rows = cur.fetchall()

            fts_candidates = [
                {
                    "source_file": r[0],
                    "page": r[1],
                    "section": r[2],
                    "content": r[3],
                    "keyword_rank": round(float(r[4]), 4)
                }
                for r in fts_rows
            ]

        # Merge using Reciprocal Rank Fusion (k=60)
        return reciprocal_rank_fusion(vector_candidates, fts_candidates, k=60)[:limit]


def get_chunk_source_key(source_file: str, section: str) -> str:
    """Formats chunk into 'source_file#sec_num' or 'source_file#section' matching expected_sources."""
    m = re.search(r"(\d+\.\d+)", section or "")
    sec_num = m.group(1) if m else (section or "")
    return f"{source_file}#{sec_num}"


def evaluate_retrieval_run(model_name: str, mode: str, questions: List[Dict[str, Any]]):
    """Evaluates hit@1, hit@3, hit@5, MRR, and target question ranks."""
    hit_1_count = 0
    hit_3_count = 0
    hit_5_count = 0
    mrr_sum = 0.0
    total_q = len(questions)
    target_ranks = {}

    for q in questions:
        qid = q["id"]
        exp_sources = q.get("expected_sources", [])
        candidates = retrieve_candidates(model_name, q["q"], mode=mode, limit=23)

        found_rank = None
        for rank_1idx, cand in enumerate(candidates, 1):
            key = get_chunk_source_key(cand["source_file"], cand["section"])
            if key in exp_sources:
                found_rank = rank_1idx
                break

        if found_rank is not None:
            if found_rank <= 1:
                hit_1_count += 1
            if found_rank <= 3:
                hit_3_count += 1
            if found_rank <= 5:
                hit_5_count += 1
            mrr_sum += 1.0 / found_rank
        else:
            mrr_sum += 0.0

        if qid in TARGET_QUESTION_IDS:
            target_ranks[qid] = found_rank

    return {
        "model": model_name,
        "mode": mode,
        "total": total_q,
        "hit@1": hit_1_count / total_q if total_q else 0.0,
        "hit@3": hit_3_count / total_q if total_q else 0.0,
        "hit@5": hit_5_count / total_q if total_q else 0.0,
        "mrr": mrr_sum / total_q if total_q else 0.0,
        "target_ranks": target_ranks,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval across embedding models and modes.")
    parser.add_argument("--models", nargs="+", default=["nomic-embed-text", "mxbai-embed-large", "bge-m3"],
                        help="List of embedding models to evaluate")
    args = parser.parse_args()

    # Load questions from dev and heldout
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, "eval", "questions.json")) as f:
        dev_questions = json.load(f)
    with open(os.path.join(base_dir, "eval", "questions_heldout.json")) as f:
        held_questions = json.load(f)

    # Filter to questions that have expected_sources and do not expect refusal
    all_eval_questions = [
        q for q in (dev_questions + held_questions)
        if q.get("expected_sources") and not q.get("expect_refusal")
    ]
    print(f"Total evaluation questions with expected sources: {len(all_eval_questions)} "
          f"({len([q for q in dev_questions if q.get('expected_sources') and not q.get('expect_refusal')])} dev + "
          f"{len([q for q in held_questions if q.get('expected_sources') and not q.get('expect_refusal')])} held-out)")

    # Check available models in Ollama
    client = get_ollama_client()
    try:
        available_models = [m.model for m in client.list().models]
    except Exception:
        available_models = []

    results = []

    for model_name in args.models:
        # Check if model is pulled
        is_pulled = any(model_name in am for am in available_models)
        if not is_pulled:
            print(f"\n[SKIP] Model '{model_name}' is not yet downloaded in Ollama. Skipping.")
            continue

        print(f"\nEvaluating model: {model_name}...")
        try:
            populate_table_if_empty(model_name)
        except Exception as e:
            print(f"[ERROR] Failed to populate index for {model_name}: {e}")
            continue

        for mode in ["vector", "hybrid"]:
            print(f"  Running retrieval ({model_name}, mode={mode})...")
            res = evaluate_retrieval_run(model_name, mode, all_eval_questions)
            results.append(res)

    # Print results summary table
    print("\n" + "=" * 80)
    print("RETRIEVAL EXPERIMENT RESULTS (Dev + Held-out, 36 Questions)")
    print("=" * 80)
    header = f"{'Model':<20} | {'Mode':<10} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'P07':<6} | {'H03':<6} | {'H08':<6}"
    print(header)
    print("-" * 80)

    for r in results:
        tr = r["target_ranks"]
        p07_str = f"#{tr.get('P07')}" if tr.get('P07') else "None"
        h03_str = f"#{tr.get('H03')}" if tr.get('H03') else "None"
        h08_str = f"#{tr.get('H08')}" if tr.get('H08') else "None"
        line = (
            f"{r['model']:<20} | {r['mode']:<10} | "
            f"{r['hit@1']*100:>5.1f}% | {r['hit@3']*100:>5.1f}% | {r['hit@5']*100:>5.1f}% | "
            f"{r['mrr']:>6.3f} | {p07_str:<6} | {h03_str:<6} | {h08_str:<6}"
        )
        print(line)
    print("=" * 80)


if __name__ == "__main__":
    main()
