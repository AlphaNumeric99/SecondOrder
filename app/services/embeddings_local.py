from __future__ import annotations

import asyncio
import hashlib
import math
import time
from typing import Any

from app.config import settings


class LocalEmbeddingService:
    """Local embedding service with graceful fallback for minimal environments."""

    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        self.model_name = model_name or settings.local_embed_model
        self.batch_size = batch_size or int(settings.local_embed_batch_size)
        self._model: Any | None = None
        self._load_attempted = False
        self._lock = asyncio.Lock()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            if not self._load_attempted:
                await asyncio.to_thread(self._load_model)
                self._load_attempted = True
        return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0]

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            self._model = None
            return
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            return [_hashed_embedding(text) for text in texts]

        retries = 3
        for attempt in range(retries):
            try:
                vectors = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return [list(map(float, row)) for row in vectors]
            except Exception:
                if attempt == retries - 1:
                    break
                time.sleep(0.2 * (attempt + 1))
        return [_hashed_embedding(text) for text in texts]


def _hashed_embedding(text: str, dim: int = 384) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [0.0] * dim
    for i in range(dim):
        b = digest[i % len(digest)]
        values[i] = (b / 127.5) - 1.0
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return values
    return [v / norm for v in values]
