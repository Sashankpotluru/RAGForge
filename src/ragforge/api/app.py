"""Local single-worker HTTP API. No authentication: bind loopback only."""

import base64
import binascii
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from ragforge.config import Settings
from ragforge.domain.models import DocumentInput, SearchRequest
from ragforge.domain.protocols import AnswerProvider, Reranker
from ragforge.evaluation.runner import index_id, run_fixture, write_artifact
from ragforge.generation.citations import build_citations, validate_citation_references
from ragforge.generation.providers import ChatCompletionsProvider
from ragforge.ingestion.loaders import load_bytes
from ragforge.ingestion.url_loader import SafeUrlLoader
from ragforge.retrieval.rerank import CrossEncoderReranker
from ragforge.retrieval.service import SearchService
from ragforge.runtime import Runtime


class DocumentPayload(DocumentInput):
    model_config = ConfigDict(extra="forbid", frozen=True)
    content: str = ""
    file_base64: str | None = None
    mime_type: str = "text/plain"


class UrlPayload(BaseModel):
    url: str
    collection: str = Field(default="default", min_length=1)
    external_id: str | None = None


class EvaluationPayload(BaseModel):
    dataset: str = Field(default="fixture", pattern="^fixture$")


def create_app(
    settings: Settings | None = None,
    *,
    runtime: Runtime | None = None,
    answer_provider: AnswerProvider | None = None,
    reranker: Reranker | None = None,
    rewriter: Callable[[str], str] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if settings.provider_base_url and settings.provider_model:
        configured = ChatCompletionsProvider(
            settings.provider_base_url,
            settings.provider_model,
            settings.provider_token.get_secret_value() if settings.provider_token else None,
            timeout=settings.request_timeout_seconds,
        )
        answer_provider = answer_provider or configured
        rewriter = rewriter or configured.rewrite

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime or Runtime(settings)
        configured_reranker = reranker
        if configured_reranker is None and settings.reranker_model:
            configured_reranker = CrossEncoderReranker(
                settings.reranker_model, settings.reranker_revision
            )
        app.state.search = SearchService(app.state.runtime, configured_reranker, rewriter)
        try:
            yield
        finally:
            if runtime is None:
                app.state.runtime.close()

    app = FastAPI(title="RAGForge", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def envelope(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = str(uuid4())
        try:
            # Bound actual streamed bytes, not merely caller-controlled Content-Length.
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > 28 * 1024 * 1024:
                    return JSONResponse(
                        {"error": "request exceeds 28 MiB", "request_id": request.state.request_id},
                        status_code=413,
                        headers={"x-request-id": request.state.request_id},
                    )
            request._body = bytes(body)
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                {"error": "internal service failure", "request_id": request.state.request_id},
                status_code=500,
            )
        response.headers["x-request-id"] = request.state.request_id
        return response

    def payload(request: Request, **values: Any) -> dict[str, Any]:
        rt: Runtime = request.app.state.runtime
        return {"request_id": request.state.request_id, "config_id": rt.config_id, **values}

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse(payload(request, error=error.detail), status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            payload(
                request,
                error="invalid request",
                details=[{"location": list(e["loc"]), "message": e["msg"]} for e in error.errors()],
            ),
            status_code=422,
        )

    @app.exception_handler(ValueError)
    async def value_error(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(payload(request, error=str(error)), status_code=400)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request: Request, error: RuntimeError) -> JSONResponse:
        return JSONResponse(payload(request, error=str(error)), status_code=503)

    @app.get("/healthz")
    def health(request: Request) -> dict[str, Any]:
        return payload(request, status="ok")

    @app.get("/readyz")
    def ready(request: Request) -> dict[str, Any]:
        try:
            if not request.app.state.runtime.dense.ready():
                raise RuntimeError("not ready")
        except Exception as error:
            raise HTTPException(503, "index dependency unavailable") from error
        return payload(request, status="ready")

    @app.post("/v1/documents")
    def ingest(document: DocumentPayload, request: Request) -> dict[str, Any]:
        loaded = None
        if document.file_base64 is not None:
            if document.content:
                raise ValueError("provide content or file_base64, not both")
            try:
                data = base64.b64decode(document.file_base64, validate=True)
            except binascii.Error as error:
                raise ValueError("invalid base64") from error
            loaded = load_bytes(data, document.mime_type, document.source)
        elif len(document.content.encode()) > 20 * 1024 * 1024:
            raise ValueError("document exceeds 20 MiB")
        normalized = DocumentInput.model_validate(
            document.model_dump(exclude={"file_base64", "mime_type"})
        )
        if loaded is not None:
            normalized = normalized.model_copy(update={"content": loaded.text})
        result = request.app.state.runtime.ingest(normalized, loaded)
        return payload(request, **asdict(result))

    @app.post("/v1/documents/url")
    def ingest_url(document: UrlPayload, request: Request) -> dict[str, Any]:
        loaded = SafeUrlLoader(settings).load(document.url)
        result = request.app.state.runtime.ingest(
            DocumentInput(
                source=loaded.source,
                content=loaded.text,
                external_id=document.external_id,
                collection=document.collection,
            ),
            loaded,
        )
        return payload(request, **asdict(result))

    @app.get("/v1/documents/{document_id}")
    def status(document_id: str, request: Request) -> dict[str, Any]:
        record = request.app.state.runtime.catalog.get_by_document_id(document_id)
        if record is None:
            raise HTTPException(404, "document not found")
        return payload(request, **record.model_dump(mode="json"))

    @app.delete("/v1/documents/{document_id}")
    def delete(document_id: str, request: Request) -> dict[str, Any]:
        request.app.state.runtime.delete(document_id)
        return payload(request, document_id=document_id, deleted=True)

    def evidence_response(search: SearchRequest, request: Request) -> dict[str, Any]:
        rt: Runtime = request.app.state.runtime
        with rt.lock:
            started = perf_counter()
            evidence = request.app.state.search.search(search)
            return payload(
                request,
                query=search.query,
                index_id=index_id(rt),
                retrieval_ms=(perf_counter() - started) * 1000,
                mode=search.mode,
                retrieval_config=search.model_dump(mode="json", exclude={"query"}),
                embedding_model=settings.embedding_model,
                embedding_revision=settings.embedding_revision,
                reranker_model=settings.reranker_model,
                reranker_revision=settings.reranker_revision,
                evidence=[r.model_dump(mode="json") for r in evidence],
                citations=[c.model_dump(mode="json") for c in build_citations(evidence)],
            )

    @app.post("/v1/search")
    def search_route(search: SearchRequest, request: Request) -> dict[str, Any]:
        return evidence_response(search, request)

    @app.post("/v1/ask")
    def ask(search: SearchRequest, request: Request) -> dict[str, Any]:
        if answer_provider is None:
            raise HTTPException(503, "answer provider is not configured; use /v1/search")
        from ragforge.domain.models import Citation, RetrievedChunk

        result = evidence_response(search, request)
        evidence = [RetrievedChunk.model_validate(r) for r in result["evidence"]]
        citations = [Citation.model_validate(c) for c in result["citations"]]
        answer = answer_provider.answer(search.query, evidence, citations)
        try:
            validate_citation_references(answer, citations)
        except ValueError as error:
            raise HTTPException(502, "provider returned invalid citation references") from error
        return {**result, "answer": answer.model_dump(mode="json")}

    @app.post("/v1/evaluations/run")
    def evaluation(body: EvaluationPayload, request: Request) -> dict[str, Any]:
        artifact = run_fixture(settings)
        path = (
            settings.data_dir
            / "evaluations"
            / f"fixture-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}.json"
        )
        write_artifact(artifact, path)
        return payload(request, artifact=artifact, artifact_path=str(path))

    return app
