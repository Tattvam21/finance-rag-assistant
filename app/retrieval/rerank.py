import math
from typing import List, Dict, Any

def rerank_chunks(query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Advanced Reranker: Takes top-N candidates (e.g. 20) and reranks down to top_k (e.g. 5).
    Combines:
    1. Dense vector cosine similarity
    2. BM25-style lexical term frequency scoring across content & section titles
    3. Exact phrase match boost for query phrases
    """
    if not candidate_chunks:
        return []

    query_tokens = [t.lower().strip("?,.:;!") for t in query.split() if len(t) > 2]
    query_phrase = query.lower()

    reranked = []
    for chunk in candidate_chunks:
        content_lower = chunk["content"].lower()
        section_lower = chunk.get("section", "").lower()

        # 1. Base vector similarity
        base_sim = float(chunk.get("similarity", 0.0))

        # 2. Lexical term match score
        matches = 0
        section_matches = 0
        for token in query_tokens:
            if token in content_lower:
                matches += 1
            if token in section_lower:
                section_matches += 1

        term_ratio = matches / len(query_tokens) if query_tokens else 0
        section_ratio = section_matches / len(query_tokens) if query_tokens else 0

        # 3. Exact phrase match (disabled for ablation)
        phrase_boost = 0.0

        # Weighted combined score
        final_score = (base_sim * 0.40) + (term_ratio * 0.35) + (section_ratio * 0.20) + phrase_boost
        
        chunk_copy = dict(chunk)
        chunk_copy["rerank_score"] = round(final_score, 4)
        chunk_copy["similarity"] = round(final_score, 4)  # Update effective similarity
        reranked.append(chunk_copy)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
