import numpy as np

from app.errors import AppError
from app.models import Passage
from app.store import VectorStore


def retrieve(
    store: VectorStore, query_vector: list[float], top_k: int, min_similarity: float
) -> list[Passage]:
    query = np.asarray(query_vector, dtype=float)
    query_norm = np.linalg.norm(query)
    if query_norm == 0 or not np.isfinite(query).all():
        raise AppError("Embedding model returned an invalid query vector.", 503)
    results = []
    for passage, vector in store.passages():
        stored = np.asarray(vector, dtype=float)
        if stored.shape != query.shape:
            raise AppError("Embedding dimensions changed. Rebuild the document index.", 503)
        norm = np.linalg.norm(stored)
        score = float(np.dot(query, stored) / (query_norm * norm)) if norm else 0
        if score >= min_similarity:
            results.append(passage.model_copy(update={"score": score}))
    return sorted(results, key=lambda passage: passage.score, reverse=True)[:top_k]
