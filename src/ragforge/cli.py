"""Local commands using the same handlers as the HTTP API."""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from fastapi.testclient import TestClient

from ragforge.api.app import create_app
from ragforge.config import Settings
from ragforge.evaluation.runner import run_fixture, write_artifact

app = typer.Typer(no_args_is_help=True)


def request(method: str, path: str, body: dict[str, Any] | None = None) -> None:
    with TestClient(create_app()) as client:
        response = client.request(method, path, json=body)
        typer.echo(json.dumps(response.json(), ensure_ascii=False))
        if response.is_error:
            raise typer.Exit(1)


@app.command()
def ingest(source: str, collection: str = "default") -> None:
    """Ingest a local file or an allowlisted HTTPS URL."""
    if source.startswith("https://"):
        request("POST", "/v1/documents/url", {"url": source, "collection": collection})
        return
    try:
        path = Path(source).resolve(strict=True)
        if not path.is_file():
            raise typer.BadParameter("expected a regular file", param_hint="SOURCE")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise typer.BadParameter("document exceeds 20 MiB", param_hint="SOURCE")
        data = path.read_bytes()
    except OSError as error:
        raise typer.BadParameter(
            f"cannot read file: {error.strerror or str(error)}", param_hint="SOURCE"
        ) from error
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    request(
        "POST",
        "/v1/documents",
        {
            "source": path.as_uri(),
            "collection": collection,
            "mime_type": mime,
            "file_base64": base64.b64encode(data).decode(),
        },
    )


@app.command()
def search(
    query: str, collection: str = "default", mode: str = "hybrid", rewrite: bool = False
) -> None:
    """Retrieve ranked evidence and citations."""
    request(
        "POST",
        "/v1/search",
        {"query": query, "collection": collection, "mode": mode, "rewrite_query": rewrite},
    )


@app.command()
def ask(query: str, collection: str = "default") -> None:
    """Generate a cited answer using the configured provider."""
    request("POST", "/v1/ask", {"query": query, "collection": collection})


@app.command()
def status(document_id: str) -> None:
    request("GET", f"/v1/documents/{document_id}")


@app.command()
def delete(document_id: str) -> None:
    request("DELETE", f"/v1/documents/{document_id}")


@app.command()
def evaluate(output: Path = Path("results/fixture.json")) -> None:
    """Write a measured fixture artifact."""
    typer.echo(str(write_artifact(run_fixture(Settings()), output)))


@app.command()
def serve(port: int = 8000) -> None:
    """Start the local single-worker API."""
    uvicorn.run("ragforge.api.app:create_app", factory=True, host="127.0.0.1", port=port)


@app.command()
def demo(
    data_dir: Annotated[Path, typer.Option()],
    port: int = 8787,
    neural: bool = False,
    generate: bool = False,
) -> None:
    """Serve the read-only sample workbench on loopback."""
    from ragforge.demo.app import create_demo_app

    uvicorn.run(
        create_demo_app(data_dir, neural, generate),
        host="127.0.0.1",
        port=port,
        proxy_headers=False,
        limit_concurrency=32,
    )
