import json

import pytest
from pydantic import ValidationError

from ragforge.config import Settings
from ragforge.domain.models import MetadataCondition
from ragforge.evaluation.metrics import retrieval_metrics
from ragforge.evaluation.runner import run_fixture


@pytest.mark.parametrize(
    ("operator", "value"), [("eq", ["a"]), ("in", 1), ("lte", "2"), ("gte", True)]
)
def test_invalid_operand_contract(operator, value):
    with pytest.raises(ValidationError):
        MetadataCondition(field="size", operator=operator, value=value)


def test_artifact_provenance_and_secret_exclusion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = run_fixture(Settings(provider_token="private-test-token"), modes=["hybrid"])
    assert artifact["source"]["kind"] in {"git-checkout", "installed-package"}
    assert artifact["source"]["version"]
    assert "private-test-token" not in json.dumps(artifact)
    assert "provider_token" not in artifact["runs"][0]["settings"]
    assert "novelty_ndcg@10" in retrieval_metrics([{0, 1}], 2)
