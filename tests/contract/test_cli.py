import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


def test_cli_ingest_search_evaluate_from_other_directory(tmp_path, monkeypatch):
    from ragforge.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAGFORGE_DATA_DIR", str(tmp_path / "runtime"))
    fixture = tmp_path / "notes.md"
    fixture.write_text("violet flowers")
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    ingested = runner.invoke(app, ["ingest", str(fixture)])
    assert ingested.exit_code == 0, ingested.output
    response = runner.invoke(app, ["search", "violet"])
    assert response.exit_code == 0, response.output
    assert json.loads(response.output)["citations"]
    evaluated = runner.invoke(app, ["evaluate", "--output", str(tmp_path / "result.json")])
    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads((tmp_path / "result.json").read_text())["runs"]


@pytest.mark.parametrize("kind", ["missing", "directory", "unreadable"])
def test_ingest_reports_unreadable_source_as_usage_error(kind, tmp_path, monkeypatch):
    from ragforge.cli import app

    path = tmp_path / "source.txt"
    if kind == "directory":
        path.mkdir()
    if kind == "unreadable":
        path.write_text("example")

        def deny_read(self):
            raise PermissionError("permission denied")

        monkeypatch.setattr(Path, "read_bytes", deny_read)
    result = CliRunner().invoke(app, ["ingest", str(path)])
    assert result.exit_code == 2
    assert "source" in result.output.lower()
    assert "Traceback" not in result.output
