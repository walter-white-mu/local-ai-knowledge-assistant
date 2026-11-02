import httpx
import numpy as np

from app.config import Settings
from app.errors import AppError


class OllamaEmbeddings:
    def __init__(self, settings: Settings):
        self.settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        try:
            with httpx.Client(timeout=self.settings.ollama_timeout) as client:
                for start in range(0, len(texts), 32):
                    batch = texts[start : start + 32]
                    response = client.post(
                        f"{self.settings.ollama_url}/api/embed",
                        json={"model": self.settings.embedding_model, "input": batch},
                    )
                    response.raise_for_status()
                    result = response.json()["embeddings"]
                    matrix = np.asarray(result, dtype=float)
                    if (
                        matrix.ndim != 2
                        or matrix.shape[0] != len(batch)
                        or matrix.shape[1] == 0
                        or not np.isfinite(matrix).all()
                        or (np.linalg.norm(matrix, axis=1) == 0).any()
                    ):
                        raise ValueError("Invalid embeddings")
                    vectors.extend(matrix.tolist())
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            raise AppError(
                "Embeddings unavailable. Check Ollama and pull the configured embedding model.", 503
            ) from exc
        return vectors
