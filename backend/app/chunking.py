import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    text: str
    section: int
    page: int | None = None


def split_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    text = re.sub(r"[ \t]+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundaries = [
                text.rfind(marker, start + size // 2, end) for marker in ("\n", ". ", " ")
            ]
            boundary = max(boundaries)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
