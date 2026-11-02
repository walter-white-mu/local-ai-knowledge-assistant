import re
from typing import Annotated

import httpx
from fastapi import APIRouter, File, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.errors import AppError
from app.generation import ABSTENTION
from app.ingestion import Ingestion
from app.models import Answer, Document, ErrorResponse, Health, Question
from app.retrieval import retrieve


def create_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=Health)
    def health(request: Request):
        try:
            response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=3)
            response.raise_for_status()
            available = {model["name"] for model in response.json()["models"]}
            missing = [
                model
                for model in (settings.embedding_model, settings.chat_model)
                if model not in available and f"{model}:latest" not in available
            ]
            ollama = "ready" if not missing else "models_missing"
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            ollama, missing = "unavailable", []
        with request.app.state.store.connect() as connection:
            connection.execute("SELECT 1")
        return Health(
            status="ok" if ollama == "ready" else "degraded",
            database="ready",
            ollama=ollama,
            missing_models=missing,
            embedding_model=settings.embedding_model,
            chat_model=settings.chat_model,
        )

    @router.get("/documents", response_model=list[Document])
    def documents(request: Request):
        return request.app.state.store.documents()

    @router.post(
        "/documents",
        response_model=Document,
        status_code=201,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def upload(request: Request, file: Annotated[UploadFile, File()]):
        filename = re.split(r"[/\\]", file.filename or "document")[-1]
        filename = "".join(character for character in filename if character.isprintable())[:200]
        try:
            content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        finally:
            await file.close()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise AppError(f"Upload exceeds the {settings.max_upload_mb} MB limit.", 413)
        state = request.app.state
        ingestion = Ingestion(settings, state.store, state.embeddings)
        return await run_in_threadpool(ingestion.ingest, filename, content)

    @router.delete("/documents/{document_id}", status_code=204)
    def delete(request: Request, document_id: str):
        request.app.state.store.delete(document_id)

    @router.post("/chat", response_model=Answer)
    def chat(request: Request, body: Question):
        question = body.question.strip()
        if not question:
            raise AppError("Enter a question.", 422)
        state = request.app.state
        if not state.store.documents():
            return Answer(answer=ABSTENTION, insufficient_evidence=True)
        query = state.embeddings.embed([question])[0]
        passages = retrieve(state.store, query, settings.top_k, settings.min_similarity)
        return state.generator.answer(question, passages)

    return router
