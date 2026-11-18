import json
import subprocess
import sys
from pathlib import Path

import pytest

from ragforge.config import Settings
from ragforge.domain.models import Chunk, DocumentInput, MetadataFilter, RetrievedChunk
from ragforge.runtime import Runtime


def test_restart_recovers_truncated_snapshot(tmp_path):
    runtime = Runtime(Settings(data_dir=tmp_path))
    runtime.ingest(DocumentInput(source="memory:a", content="violet"))
    runtime.close()
    (tmp_path / "bm25.json").write_text("{")
    runtime = Runtime(Settings(data_dir=tmp_path))
    assert runtime.lexical.search("violet", MetadataFilter(), 1)
    runtime.close()


def test_rewrite_provenance_does_not_overwrite():
    from ragforge.retrieval.fusion import reciprocal_rank_fusion

    chunk = Chunk(
        chunk_id="id", document_id="d", version="v", ordinal=0, text="violet", source_location="s"
    )
    first = RetrievedChunk(chunk=chunk, rank=1, scores={"dense": 0.9}, source_ranks={"dense": 1})
    revised = first.model_copy(update={"scores": {"dense": 0.5}})
    result = reciprocal_rank_fusion({"dense": [first], "dense_rewritten": [revised]})[0]
    assert result.scores["dense"] == 0.9
    assert result.scores["dense_rewritten"] == 0.5


def test_binary_ndcg_for_unique_chunk_ids():
    from ragforge.evaluation.metrics import binary_ndcg

    # Two relevant chunks exist; a appears twice, b is missed.
    assert binary_ndcg(["none", "a", "a"], {"a", "b"}) == pytest.approx(0.6309297536 / 1.6309297536)
    assert binary_ndcg(["a"], set()) == 0


def test_cli_stdout_is_one_json_document(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGFORGE_DATA_DIR", str(tmp_path))
    cli = Path(sys.executable).with_name("ragforge")
    result = subprocess.run([cli, "search", "violet"], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["evidence"] == []
