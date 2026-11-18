import pytest


def test_distinct_span_recall_hit_rate_and_duplicate_discount() -> None:
    from ragforge.evaluation.metrics import retrieval_metrics

    # Three returned chunks, first two overlap the SAME gold span; second span is missed.
    result = retrieval_metrics([{0}, {0}, set()], total_spans=2)
    assert result["recall@1"] == 0.5
    assert result["hit_rate@1"] == 1.0
    assert result["recall@3"] == 0.5
    assert result["mrr@10"] == 1.0
    assert result["novelty_ndcg@10"] == pytest.approx(1 / (1 + 1 / 1.584962500721156))
    assert retrieval_metrics([set()], 2)["novelty_ndcg@10"] == 0.0


def test_fixture_runner_produces_reproducible_metadata(tmp_path) -> None:
    from ragforge.config import Settings
    from ragforge.evaluation.runner import run_fixture

    artifact = run_fixture(Settings(data_dir=tmp_path), modes=["dense", "hybrid"])
    assert artifact["dataset_id"] == "ragforge-original-fixture-v1"
    assert len(artifact["runs"]) == 2
    assert artifact["hardware"] and artifact["dependencies"] and artifact["git_commit"]
    for run in artifact["runs"]:
        assert run["cases"] and run["metrics"]["recall@10"] >= 0
        assert run["latency_ms"]["p95"] >= 0
        assert run["index_id"]
