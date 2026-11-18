"""Read-only application over an isolated, ephemeral index of public sample text."""

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import monotonic
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from starlette.concurrency import run_in_threadpool

from ragforge.config import Settings
from ragforge.demo.local import MODEL, REVISION, LocalGenerator
from ragforge.domain.models import DocumentInput, SearchRequest
from ragforge.generation.citations import build_citations
from ragforge.retrieval.rerank import CrossEncoderReranker
from ragforge.retrieval.service import SearchService
from ragforge.runtime import Runtime

ROOT = Path(__file__).parent
EMBED = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
EMBED_REV = "f3cb857cba53019a20df283396bcca179cf051a4"
RERANK = "cross-encoder/ms-marco-TinyBERT-L2-v2"
RERANK_REV = "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc"


class DemoSettings(Settings):
    """The sample runtime accepts settings exclusively from the demo factory."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid"


def create_demo_app(data_dir: Path, neural: bool = False, generate: bool = False) -> FastAPI:
    fixture = json.loads((ROOT.parent / "evaluation/fixture.json").read_text())
    generator = LocalGenerator() if generate else None
    generation_lock = Lock()
    queries: deque[float] = deque()
    active = 0
    modes = ["dense", "hybrid"] + (["hybrid_rerank"] if neural else [])

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        data_dir.mkdir(parents=True, exist_ok=True)
        if any(data_dir.iterdir()):
            raise ValueError("demo requires an empty dedicated directory")
        with TemporaryDirectory(dir=data_dir, prefix="sample-") as storage:
            settings = DemoSettings(
                data_dir=Path(storage),
                embedding_model=EMBED if neural else "hash-demo-v1",
                embedding_revision=EMBED_REV if neural else None,
            )
            runtime = Runtime(settings)
            try:
                for doc in fixture["documents"]:
                    runtime.ingest(DocumentInput(**doc, collection="sample"))
                reranker = CrossEncoderReranker(RERANK, RERANK_REV) if neural else None
                app.state.search = SearchService(runtime, reranker=reranker)
                yield
            finally:
                runtime.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def headers(request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers.update(
            {
                "Content-Security-Policy": (
                    "default-src 'self'; object-src 'none'; "
                    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
                ),
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            }
        )
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(ROOT / "static/index.html")

    @app.get("/assets/{asset}")
    def static(
        asset: Literal["workbench.js", "workbench.css", "architecture-evaluation.png"],
    ) -> FileResponse:
        return FileResponse(ROOT / "static" / asset)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "modes": modes,
            "embedding": EMBED if neural else "hash-demo-v1",
            "generation": MODEL if generator else None,
            "generation_revision": REVISION if generator else None,
        }

    @app.get("/api/corpus")
    def corpus() -> dict[str, Any]:
        return {
            "documents": fixture["documents"],
            "questions": [case["question"] for case in fixture["cases"]],
        }

    @app.get("/api/results")
    def results() -> FileResponse:
        return FileResponse(ROOT / "measurements.json")

    def execute(query: Query, answer: bool) -> dict[str, Any]:
        started = monotonic()
        chunks = app.state.search.search(
            SearchRequest(query=query.query, mode=query.mode, collection="sample")
        )
        citations = build_citations(chunks)
        result = {
            "hits": [
                dict(c.model_dump(), title=r.chunk.title, scores=r.scores, rank=r.rank)
                for c, r in zip(citations, chunks, strict=True)
            ],
            "search_ms": round((monotonic() - started) * 1000, 2),
        }
        if answer and generator:
            started = monotonic()
            try:
                result["answer"] = generator.answer(query.query, citations)
            except Exception:
                result["generation_error"] = (
                    "Local generation could not produce a validated cited answer."
                )
            result["generation_ms"] = round((monotonic() - started) * 1000, 2)
        return result

    @app.post("/api/search")
    @app.post("/api/answer")
    async def handle(request: Request) -> Response:
        nonlocal active
        now = monotonic()
        while queries and now - queries[0] >= 60:
            queries.popleft()
        if len(queries) >= 30 or active >= 3:
            raise HTTPException(429, "Demo busy. Try again shortly.")
        answer = request.url.path == "/api/answer"
        if answer and not generator:
            raise HTTPException(503, "Local generation is disabled.")
        if answer and not generation_lock.acquire(blocking=False):
            raise HTTPException(429, "Generation is busy.")
        active += 1
        queries.append(now)
        try:
            raw = bytearray()
            try:
                async with asyncio.timeout(5):
                    async for chunk in request.stream():
                        if len(raw) + len(chunk) > 8192:
                            raise HTTPException(413, "Request exceeds 8 KiB.")
                        raw.extend(chunk)
            except TimeoutError:
                raise HTTPException(408, "Request body timed out.") from None
            try:
                query = Query.model_validate_json(raw)
            except ValidationError:
                raise HTTPException(
                    422, "Expected a question of 1–500 characters and a supported mode."
                ) from None
            if query.mode not in modes:
                raise HTTPException(422, "Mode unavailable.")
            return JSONResponse(await run_in_threadpool(execute, query, answer))
        finally:
            active -= 1
            if answer:
                generation_lock.release()

    return app
