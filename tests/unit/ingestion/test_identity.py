def test_canonical_source_removes_fragment_and_normalizes_url_host() -> None:
    from ragforge.ingestion.identity import canonical_source

    source = canonical_source("HTTPS://EXAMPLE.COM/policies/security.html?revision=2#overview")

    assert source == "https://example.com/policies/security.html?revision=2"


def test_content_hash_is_repeatable_and_sensitive_to_content() -> None:
    from ragforge.ingestion.identity import content_hash

    assert content_hash("same content") == content_hash("same content")
    assert content_hash("same content") != content_hash("changed content")
