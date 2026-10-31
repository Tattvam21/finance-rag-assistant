import time
import logging
from typing import List, Tuple, Optional
from app.config import EMBED_MODEL, EMBED_DIM, get_ollama_client

logger = logging.getLogger("finance_rag.embed")

QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "

EMBEDDER_SPECS = {
    "nomic-embed-text": {
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "dim": 768,
        "table": "chunks",
    },
    "mxbai-embed-large": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "dim": 1024,
        "table": "chunks_mxbai",
    },
    "bge-m3": {
        "query_prefix": "",
        "doc_prefix": "",
        "dim": 1024,
        "table": "chunks_bge",
    },
}

def get_embedder_spec(model_name: Optional[str] = None) -> dict:
    target = (model_name or EMBED_MODEL).lower()
    for k, v in EMBEDDER_SPECS.items():
        if k in target or target in k:
            return v
    return EMBEDDER_SPECS["nomic-embed-text"]

def embed_text(text: str, prefix: Optional[str] = None, model: Optional[str] = None) -> Tuple[List[float], float]:
    """
    Embeds a single string with the model-specific task prefix.
    Returns:
        (embedding_vector, latency_ms)
    """
    target_model = model or EMBED_MODEL
    spec = get_embedder_spec(target_model)
    pref = prefix if prefix is not None else spec["query_prefix"]
    clean_text = text.strip()
    prefixed_input = f"{pref}{clean_text}" if (pref and not clean_text.startswith(pref)) else clean_text

    client = get_ollama_client()
    t0 = time.time()
    try:
        resp = client.embed(model=target_model, input=prefixed_input)
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate embedding with Ollama model '{target_model}': {e}\n"
            f"Ensure Ollama is running and run 'ollama pull {target_model}'."
        ) from e

    latency_ms = round((time.time() - t0) * 1000, 2)
    if not resp.embeddings or len(resp.embeddings) == 0:
        raise RuntimeError(f"Ollama returned an empty embedding response for model '{target_model}'.")

    emb = resp.embeddings[0]
    expected_dim = spec["dim"]
    if len(emb) != expected_dim:
        logger.warning(f"Embedding dimension {len(emb)} differs from configured spec dim={expected_dim}")
    return emb, latency_ms

def get_query_embedding(query: str, model: Optional[str] = None) -> Tuple[List[float], float]:
    """Generates embedding for a search query using the model's query prefix."""
    return embed_text(query, prefix=None, model=model)

def get_embedding(query: str) -> List[float]:
    """Backward compatibility helper returning embedding vector."""
    emb, _ = get_query_embedding(query)
    return emb

def get_document_embeddings(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    Generates embeddings for document chunks using the 'search_document: ' prefix.
    Batches inputs using client.embed(model=EMBED_MODEL, input=[...]).
    """
    client = get_ollama_client()
    all_embeddings: List[List[float]] = []

    # Apply nomic prefix to each document chunk
    prefixed_texts = [
        f"{DOCUMENT_PREFIX}{t.strip()}" if not t.strip().startswith(DOCUMENT_PREFIX) else t.strip()
        for t in texts
    ]

    for i in range(0, len(prefixed_texts), batch_size):
        batch = prefixed_texts[i : i + batch_size]
        try:
            resp = client.embed(model=EMBED_MODEL, input=batch)
            if not resp.embeddings or len(resp.embeddings) != len(batch):
                raise RuntimeError(
                    f"Batch embed returned {len(resp.embeddings) if resp.embeddings else 0} embeddings "
                    f"for batch size {len(batch)}"
                )
            all_embeddings.extend(resp.embeddings)
        except Exception as e:
            raise RuntimeError(
                f"Failed batch embedding with Ollama model '{EMBED_MODEL}' at batch index {i}: {e}\n"
                f"Ensure Ollama is active and model is pulled: 'ollama pull {EMBED_MODEL}'"
            ) from e

    return all_embeddings

def get_embeddings(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """Backward compatibility helper."""
    return get_document_embeddings(texts, batch_size=batch_size)
