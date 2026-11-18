def test_chunker_skips_whitespace_and_keeps_stable_nonempty_chunks() -> None:
    from ragforge.ingestion.chunker import SemanticChunker
    from ragforge.ingestion.loaders import LoadedDocument, LoadedSegment

    document = LoadedDocument(
        source="file:///knowledge/security.md",
        segments=(
            LoadedSegment(text="  \n\t", source_location="file:///knowledge/security.md"),
            LoadedSegment(
                text="One useful sentence. Another useful sentence. A final useful sentence.",
                source_location="file:///knowledge/security.md",
            ),
        ),
    )

    chunks = SemanticChunker(target_chars=28, overlap_chars=6).chunk(
        document, document_id="doc-1", version="version-1"
    )

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text.strip() for chunk in chunks)
    assert len({(chunk.start_offset, chunk.end_offset) for chunk in chunks}) == len(chunks)
    assert all(chunk.source_location == "file:///knowledge/security.md" for chunk in chunks)


def test_chunker_bounds_unsplittable_sentences() -> None:
    from ragforge.ingestion.chunker import SemanticChunker
    from ragforge.ingestion.loaders import LoadedDocument, LoadedSegment

    text = "This single sentence is deliberately longer than the configured target."
    document = LoadedDocument(
        source="file:///knowledge/long.md",
        segments=(LoadedSegment(text=text, source_location="file:///knowledge/long.md"),),
    )

    chunks = SemanticChunker(target_chars=20, overlap_chars=5).chunk(
        document, document_id="doc-2", version="version-1"
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 20 for chunk in chunks)
    assert chunks[-1].end_offset == len(text)


def test_short_prefix_and_long_token_keep_bounded_exact_offsets():
    from ragforge.ingestion.chunker import SemanticChunker
    from ragforge.ingestion.loaders import LoadedDocument, LoadedSegment

    text = "Short. " + "x" * 100_000
    chunks = SemanticChunker().chunk(
        LoadedDocument("memory:a", (LoadedSegment(text, "memory:a"),)), "d", "v"
    )
    assert chunks[-1].end_offset == len(text)
    assert all(len(c.text) <= 800 and text[c.start_offset : c.end_offset] == c.text for c in chunks)


def test_chunk_offsets_map_to_normalized_source() -> None:
    from ragforge.ingestion.chunker import SemanticChunker
    from ragforge.ingestion.loaders import LoadedDocument, LoadedSegment

    text = "First sentence. Second sentence. Third sentence."
    chunks = SemanticChunker(25, 5).chunk(
        LoadedDocument("memory:a", (LoadedSegment(text, "memory:a"),)), "d", "v"
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert text[chunk.start_offset : chunk.end_offset] == chunk.text
    assert chunks[-1].start_offset > 0
