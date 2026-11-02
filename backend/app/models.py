from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    filename: str
    sections: int
    chunks: int
    status: Literal["ready"] = "ready"
    created_at: str


class Passage(BaseModel):
    id: str
    document_id: str
    filename: str
    page: int | None = None
    section: int
    text: str
    score: float = 0


class Citation(Passage):
    label: int


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False


class ErrorResponse(BaseModel):
    detail: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ready"]
    ollama: str
    missing_models: list[str]
    embedding_model: str
    chat_model: str
