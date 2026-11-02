import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings
from app.embeddings import OllamaEmbeddings
from app.errors import AppError
from app.generation import OllamaGenerator
from app.limits import RequestBodyLimit
from app.routes import create_router
from app.store import VectorStore

logger = logging.getLogger("knowledge_assistant")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = VectorStore(settings.data_dir / "knowledge.db", settings.embedding_model)
        app.state.embeddings = OllamaEmbeddings(settings)
        app.state.generator = OllamaGenerator(settings)
        yield

    app = FastAPI(title="Local AI Knowledge Assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestBodyLimit, upload_bytes=settings.max_upload_mb * 1024 * 1024)

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        request_id = str(uuid4())
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            )
        )
        return response

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.include_router(create_router(settings))
    return app


app = create_app()
