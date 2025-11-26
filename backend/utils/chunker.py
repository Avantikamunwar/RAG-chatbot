from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping word chunks for embedding."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
    return chunks
