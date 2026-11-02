from datetime import UTC, datetime
from uuid import uuid4

from app.chunking import split_text
from app.config import Settings
from app.embeddings import OllamaEmbeddings
from app.extraction import extract_sections
from app.models import Document, Passage
from app.store import VectorStore


class Ingestion:
    def __init__(self, settings: Settings, store: VectorStore, embeddings: OllamaEmbeddings):
        self.settings = settings
        self.store = store
        self.embeddings = embeddings

    def ingest(self, filename: str, content: bytes) -> Document:
        sections = extract_sections(filename, content, self.settings.max_document_chars)
        document_id = str(uuid4())
        passages = [
            Passage(
                id=str(uuid4()),
                document_id=document_id,
                filename=filename,
                page=section.page,
                section=section.section,
                text=text,
            )
            for section in sections
            for text in split_text(
                section.text, self.settings.chunk_size, self.settings.chunk_overlap
            )
        ]
        vectors = self.embeddings.embed([passage.text for passage in passages])
        document = Document(
            id=document_id,
            filename=filename,
            sections=len(sections),
            chunks=len(passages),
            created_at=datetime.now(UTC).isoformat(),
        )
        self.store.add(document, passages, vectors)
        return document
