from pathlib import Path

import pymupdf as fitz
import pytest


@pytest.mark.parametrize("fixture_name", ["sample.md", "sample.html"])
def test_loads_textual_fixture_with_source_location(fixture_name: str) -> None:
    from ragforge.ingestion.loaders import load_local_document

    fixture = Path("data/fixtures") / fixture_name
    document = load_local_document(fixture)

    assert document.source == fixture.resolve().as_uri()
    assert document.segments[0].text.strip()
    assert document.segments[0].source_location == fixture.resolve().as_uri()


def test_loads_pdf_with_page_level_source_location(tmp_path: Path) -> None:
    from ragforge.ingestion.loaders import load_local_document

    path = tmp_path / "benefits.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Benefits enrollment closes on Friday.")
    pdf.save(path)
    pdf.close()

    document = load_local_document(path)

    assert document.segments[0].text == "Benefits enrollment closes on Friday."
    assert document.segments[0].source_location.endswith("#page=1")


def test_rejects_unsupported_local_document(tmp_path: Path) -> None:
    from ragforge.ingestion.loaders import IngestionError, load_local_document

    unsupported = tmp_path / "notes.docx"
    unsupported.write_text("not a docx")

    with pytest.raises(IngestionError, match="unsupported"):
        load_local_document(unsupported)
