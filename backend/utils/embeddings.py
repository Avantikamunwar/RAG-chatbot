from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import requests

from backend.config import get_settings


@lru_cache(maxsize=1)
def _settings() -> dict:
    return get_settings()


def _ollama_embeddings(text: str) -> List[float]:
    settings = _settings()
    payload = {
        "model": settings["ollama_embed_model"],
        "prompt": text,
    }
    url = f"{settings['ollama_base_url']}/api/embeddings"
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to reach Ollama embeddings endpoint at {url}: {exc}"
        ) from exc

    embedding = data.get("embedding")
    if not embedding:
        raise RuntimeError(
            f"Ollama embeddings response missing 'embedding': {data}"
        )
    return embedding


class OllamaEmbeddings:
    def __init__(self, model: str | None = None):
        self.model = model or _settings()["ollama_embed_model"]

    def embed_query(self, text: str) -> List[float]:
        clean_text = text.replace("\n", " ")
        return _ollama_embeddings(clean_text)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
