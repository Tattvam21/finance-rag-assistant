import time
import json
import math
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.db import get_connection
from app.ingestion.embed import get_query_embedding, get_embedder_spec
from app.config import CHAT_MODEL, EMBED_MODEL, get_ollama_client

logger = logging.getLogger("finance_rag.search")

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    fts_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merges vector candidates and full-text search candidates using Reciprocal Rank Fusion (k=60).
    RRF score = 1 / (k + rank_vector) + 1 / (k + rank_fts).
    Preserves original vector similarity for threshold guarding.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Rank vector results (1-indexed)
    for rank, doc in enumerate(vector_results, 1):
        key = f"{doc['source_file']}:{doc['page']}:{doc['section']}"
        scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in doc_map:
            doc_copy = dict(doc)
            doc_copy["vector_similarity"] = doc.get("similarity", 0.0)
            doc_map[key] = doc_copy
        doc_map[key]["vector_rank"] = rank

    # Rank FTS results (1-indexed)
    for rank, doc in enumerate(fts_results, 1):
        key = f"{doc['source_file']}:{doc['page']}:{doc['section']}"
        scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in doc_map:
            doc_copy = dict(doc)
            doc_copy["vector_similarity"] = 0.0
            doc_map[key] = doc_copy
        doc_map[key]["keyword_rank"] = rank

    # Sort merged results by RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    merged = []
    for k_item in sorted_keys:
        d = doc_map[k_item]
        d["rrf_score"] = round(scores[k_item], 6)
        if "vector_rank" not in d:
            d["vector_rank"] = None
        if "keyword_rank" not in d:
            d["keyword_rank"] = None
        # Ensure similarity reflects vector similarity for threshold checks
        if "vector_similarity" in d:
            d["similarity"] = d["vector_similarity"]
        merged.append(d)

    return merged

def pointwise_ollama_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    model: str = CHAT_MODEL
) -> List[Dict[str, Any]]:
    """
    Pointwise LLM reranking: Prompts Ollama to rate each chunk 0-10 for query relevance.
    """
    client = get_ollama_client()
    scored_candidates = []

    for c in candidates[:15]:  # Rerank top 15 to stay fast
        prompt = (
            f"Query: {query}\n\n"
            f"Document Chunk from {c['source_file']} (§{c['section']}):\n{c['content']}\n\n"
            f"Rate how directly and accurately this chunk answers the query on an integer scale from 0 to 10.\n"
            f"Return ONLY valid JSON matching: {{\"score\": <0-10>}}"
        )
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format={"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]},
                options={"temperature": 0.0, "num_ctx": 2048},
                think=False
            )
            raw = resp.message.content
            data = json.loads(raw)
            score = float(data.get("score", 0.0))
        except Exception:
            score = c.get("similarity", 0.0) * 10

        c_copy = dict(c)
        c_copy["rerank_score"] = score
        scored_candidates.append(c_copy)

    scored_candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return scored_candidates[:top_k]

def search_chunks_with_timing(
    query: str,
    doc_type: Optional[str] = None,
    limit: int = 5,
    retrieval_mode: str = "hybrid",
    use_reranker: bool = False,
    backend: Optional[str] = None,
    embed_model: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Executes chunk search with support for:
      - mode='hybrid': Combines vector search + FTS merged with RRF (k=60)
      - mode='vector': Pure vector cosine distance search
      - optional pointwise LLM reranker (--rerank)
    Returns: (chunks, embed_time_ms, retrieval_time_ms)
    """
    # 1. Generate query embedding with model prefix and resolve target table
    target_embed_model = embed_model or EMBED_MODEL
    spec = get_embedder_spec(target_embed_model)
    table_name = spec["table"]
    query_vec, embed_time_ms = get_query_embedding(query, model=target_embed_model)

    t_ret_start = time.time()
    conn, engine = get_connection(backend=backend)
    cur = conn.cursor()

    candidate_limit = 20
    vector_candidates: List[Dict[str, Any]] = []
    fts_candidates: List[Dict[str, Any]] = []

    # ----------------- VECTOR RETRIEVAL -----------------
    if engine == "postgres":
        if doc_type:
            sql_vec = f"""
                SELECT content, source_file, page, section,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {table_name}
                WHERE doc_type = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cur.execute(sql_vec, (query_vec, doc_type, query_vec, candidate_limit))
        else:
            sql_vec = f"""
                SELECT content, source_file, page, section,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cur.execute(sql_vec, (query_vec, query_vec, candidate_limit))

        rows = cur.fetchall()
        vector_candidates = [
            {
                "content": r[0],
                "source_file": r[1],
                "page": r[2],
                "section": r[3],
                "similarity": round(float(r[4]), 4)
            }
            for r in rows
        ]

        # ----------------- POSTGRES FTS RETRIEVAL -----------------
        if retrieval_mode == "hybrid":
            words = [w for w in query.split() if w.isalnum()]
            if words:
                try:
                    sql_and = f"""
                        SELECT content, source_file, page, section,
                               ts_rank(to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')), plainto_tsquery('english', %s)) AS rank
                        FROM {table_name}
                        WHERE to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')) @@ plainto_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s;
                    """
                    clean_q = " ".join(words)
                    cur.execute(sql_and, (clean_q, clean_q, candidate_limit))
                    fts_rows = cur.fetchall()

                    # Fallback to OR query if phrase / AND query returned no matches
                    if not fts_rows and len(words) > 1:
                        or_query = " | ".join(words)
                        sql_or = f"""
                            SELECT content, source_file, page, section,
                                   ts_rank(to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')), to_tsquery('english', %s)) AS rank
                            FROM {table_name}
                            WHERE to_tsvector('english', coalesce(section, '') || ' ' || coalesce(content, '')) @@ to_tsquery('english', %s)
                            ORDER BY rank DESC
                            LIMIT %s;
                        """
                        cur.execute(sql_or, (or_query, or_query, candidate_limit))
                        fts_rows = cur.fetchall()

                    fts_candidates = [
                        {
                            "content": r[0],
                            "source_file": r[1],
                            "page": r[2],
                            "section": r[3],
                            "fts_rank": float(r[4])
                        }
                        for r in fts_rows
                    ]
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    logger.debug(f"Postgres FTS query skipped: {e}")

    else:
        # SQLite storage
        tbl_name = "chunks_bge" if "bge" in target_embed_model else ("chunks_mxbai" if "mxbai" in target_embed_model else "chunks")
        if doc_type:
            cur.execute(f"SELECT content, source_file, page, section, embedding FROM {tbl_name} WHERE doc_type = ?;", (doc_type,))
        else:
            cur.execute(f"SELECT content, source_file, page, section, embedding FROM {tbl_name};")

        rows = cur.fetchall()
        scored = []
        for r in rows:
            content, source_file, page, section, emb_raw = r[0], r[1], r[2], r[3], r[4]
            emb = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
            sim = cosine_similarity(query_vec, emb)
            scored.append({
                "content": content,
                "source_file": source_file,
                "page": page,
                "section": section,
                "similarity": round(sim, 4)
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        vector_candidates = scored[:candidate_limit]

        # SQLite keyword / FTS retrieval
        if retrieval_mode == "hybrid":
            words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", query) if len(w) >= 2 and w.lower() not in ("what", "when", "does", "with", "from", "that")]
            fts_matches = []
            for r in scored:
                text_low = (r["section"] + " " + r["content"]).lower()
                matches = sum(1 for w in words if w in text_low)
                if matches > 0:
                    fts_copy = dict(r)
                    fts_copy["keyword_matches"] = matches
                    fts_matches.append(fts_copy)
            fts_matches.sort(key=lambda x: x.get("keyword_matches", 0), reverse=True)
            fts_candidates = fts_matches[:candidate_limit]

    conn.close()

    # Merge via RRF if hybrid mode
    if retrieval_mode == "hybrid":
        candidates = reciprocal_rank_fusion(vector_candidates, fts_candidates, k=60)
    else:
        candidates = []
        for rank, doc in enumerate(vector_candidates, 1):
            d = dict(doc)
            d["vector_rank"] = rank
            d["keyword_rank"] = None
            candidates.append(d)

    # Pointwise LLM Reranking (if enabled)
    if use_reranker and candidates:
        final_chunks = pointwise_ollama_rerank(query, candidates, top_k=limit)
    else:
        final_chunks = candidates[:limit]

    retrieval_time_ms = round((time.time() - t_ret_start) * 1000, 2)
    return final_chunks, embed_time_ms, retrieval_time_ms

def search_chunks(
    query: str,
    doc_type: Optional[str] = None,
    limit: int = 5,
    retrieval_mode: str = "hybrid",
    use_reranker: bool = False,
    backend: Optional[str] = None
) -> List[Dict[str, Any]]:
    chunks, _, _ = search_chunks_with_timing(
        query=query,
        doc_type=doc_type,
        limit=limit,
        retrieval_mode=retrieval_mode,
        use_reranker=use_reranker,
        backend=backend
    )
    return chunks
