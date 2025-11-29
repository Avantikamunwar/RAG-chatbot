from functools import lru_cache

import requests

from backend.config import get_settings
from backend.utils.chunker import chunk_text
from backend.utils.embeddings import OllamaEmbeddings
from backend.utils.loaders import load_all_pdfs, load_all_texts
from backend.utils.pinecone_client import get_index, insert_vectors

SIMILARITY_THRESHOLD = 0.75

@lru_cache(maxsize=1)
def _embedding_client() -> OllamaEmbeddings:
    return OllamaEmbeddings()


def _ollama_chat(prompt: str) -> str:
    settings = get_settings()
    url = f"{settings['ollama_base_url']}/api/chat"
    payload = {
        "model": settings["ollama_chat_model"],
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to reach Ollama chat endpoint at {url}: {exc}"
        ) from exc

    message = data.get("message") or {}
    content = message.get("content") or data.get("response")
    if not content:
        raise RuntimeError(
            f"Ollama chat response missing content: {data}"
        )
    return content


def build_vector_db() -> str:
    """Ingest all PDF/TXT files into Pinecone."""
    documents = {**load_all_pdfs(), **load_all_texts()}
    if not documents:
        raise RuntimeError("No documents found in data/ directory.")

    index = get_index()
    for name, text in documents.items():
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = _embedding_client().embed_documents(chunks)
        insert_vectors(
            index,
            embeddings,
            [{"text": chunk, "source": name} for chunk in chunks],
            id_prefix=name,
        )
    return f"Ingested {len(documents)} documents."


def retrieve(query: str, top_k: int = 3) -> str:
    index = get_index()
    q_emb = _embedding_client().embed_query(query)
    res = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )
    matches = res.get("matches") or []
    filtered = [
        m for m in matches
        if m.get("score", 0) >= SIMILARITY_THRESHOLD
    ]
    if not filtered:
        return ""
    context = "\n".join(
        [
            match["metadata"]["text"]
            for match in filtered
            if match.get("metadata") and match["metadata"].get("text")
        ]
    )
    return context


def generate_answer(query: str) -> str:
    context = retrieve(query)
    if not context:
        return "I don't know."

    prompt = f"""You are a helpful RAG AI assistant.

Context:
{context}

Question:
{query}

Answer using only the context. If answer not found, say "I don't know."
"""
    return _ollama_chat(prompt)
