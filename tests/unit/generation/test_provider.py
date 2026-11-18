import httpx
import pytest

from ragforge.domain.models import Chunk, RetrievedChunk
from ragforge.generation.citations import build_citations


def test_http_provider_structured_answer_and_rewrite():
    from ragforge.generation.providers import ChatCompletionsProvider

    evidence = [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="a",
                document_id="d",
                version="v",
                ordinal=0,
                text="Violet",
                source_location="memory:a",
            ),
            rank=1,
        )
    ]
    seen = []

    def respond(request):
        import json

        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["authorization"] == "Bearer test-secret"
        content = (
            '{"query":"violet flowers"}'
            if "rewrite" in body["messages"][0]["content"]
            else '{"text":"Violet [C1]","citation_ids":["C1"]}'
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = ChatCompletionsProvider(
        "https://llm.example/v1", "model", "test-secret", client=client
    )
    assert provider.answer("what?", evidence, build_citations(evidence)).citation_ids == ("C1",)
    assert provider.rewrite("violet") == "violet flowers"
    assert len(seen) == 2
    client.close()


def test_http_provider_rejects_malformed_or_failed_response():
    from ragforge.generation.providers import ChatCompletionsProvider

    for response in [httpx.Response(500), httpx.Response(200, json={"choices": []})]:
        with httpx.Client(
            transport=httpx.MockTransport(lambda r, response=response: response)
        ) as client:
            provider = ChatCompletionsProvider(
                "https://llm.example/v1", "m", "secret", client=client
            )
            with pytest.raises(RuntimeError, match="provider"):
                provider.rewrite("query")
