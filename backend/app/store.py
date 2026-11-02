import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.errors import AppError
from app.models import Document, Passage


class VectorStore:
    def __init__(self, path: Path, embedding_model: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, sections INTEGER NOT NULL,
                    chunks INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id)
                    ON DELETE CASCADE, filename TEXT NOT NULL, page INTEGER,
                    section INTEGER NOT NULL,
                    text TEXT NOT NULL, vector TEXT NOT NULL
                );
            """)
            previous = connection.execute(
                "SELECT value FROM metadata WHERE key = 'embedding_model'"
            ).fetchone()
            if previous and previous[0] != embedding_model:
                raise RuntimeError(
                    "Embedding model changed. Use a new DATA_DIR or rebuild the index."
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('embedding_model', ?)", (embedding_model,)
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def add(self, document: Document, passages: list[Passage], vectors: list[list[float]]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (
                    document.id,
                    document.filename,
                    document.sections,
                    document.chunks,
                    document.created_at,
                ),
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        passage.id,
                        passage.document_id,
                        passage.filename,
                        passage.page,
                        passage.section,
                        passage.text,
                        json.dumps(vector),
                    )
                    for passage, vector in zip(passages, vectors, strict=True)
                ],
            )

    def documents(self) -> list[Document]:
        with self.connect() as connection:
            return [
                Document(**dict(row))
                for row in connection.execute("SELECT * FROM documents ORDER BY created_at DESC")
            ]

    def passages(self) -> list[tuple[Passage, list[float]]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM chunks").fetchall()
        result = []
        for row in rows:
            values = dict(row)
            vector = json.loads(values.pop("vector"))
            result.append((Passage(**values), vector))
        return result

    def delete(self, document_id: str) -> None:
        with self.connect() as connection:
            deleted = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            if deleted.rowcount == 0:
                raise AppError("Document not found.", 404)
