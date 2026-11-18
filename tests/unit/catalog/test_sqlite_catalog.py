from pathlib import Path


def _document(source: str, content: str):
    from ragforge.domain.models import DocumentInput

    return DocumentInput(source=source, content=content, external_id="employee-handbook")


def test_catalog_detects_new_noop_and_replacement(tmp_path: Path) -> None:
    from ragforge.catalog.sqlite import SqliteCatalog
    from ragforge.domain.models import IngestDecision
    from ragforge.ingestion.identity import content_hash

    catalog = SqliteCatalog(tmp_path / "catalog.sqlite3")
    original = _document("file:///knowledge/handbook.md", "version one")

    assert catalog.register(original, content_hash(original.content)) is IngestDecision.NEW
    stored = catalog.get_by_identity("default", "external:employee-handbook")
    catalog.promote(stored.document_id, content_hash(original.content))
    assert catalog.register(original, content_hash(original.content)) is IngestDecision.NOOP

    moved = _document("file:///archive/handbook.md", "version two")
    assert catalog.register(moved, content_hash(moved.content)) is IngestDecision.REPLACE

    stored = catalog.get_by_identity("default", "external:employee-handbook")
    assert stored is not None
    assert stored.source == "file:///archive/handbook.md"


def test_catalog_persists_logical_identity_across_restart(tmp_path: Path) -> None:
    from ragforge.catalog.sqlite import SqliteCatalog
    from ragforge.domain.models import IngestDecision
    from ragforge.ingestion.identity import content_hash

    db_path = tmp_path / "catalog.sqlite3"
    first_catalog = SqliteCatalog(db_path)
    record = _document("file:///knowledge/handbook.md", "version one")
    assert first_catalog.register(record, content_hash(record.content)) is IngestDecision.NEW

    restarted_catalog = SqliteCatalog(db_path)
    moved_record = _document("file:///new-location/handbook.md", "version two")
    decision = restarted_catalog.register(moved_record, content_hash(moved_record.content))
    assert decision is IngestDecision.REPLACE
    assert (
        restarted_catalog.get_by_identity("default", "external:employee-handbook").document_id
        == first_catalog.get_by_identity("default", "external:employee-handbook").document_id
    )
