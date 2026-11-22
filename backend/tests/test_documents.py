import pytest
from io import BytesIO
from pypdf import PdfWriter

from app.documents import Section, chunk_sections, extract_sections, ingest_document


def test_txt_extracts_utf8_text():
    assert extract_sections("note.txt", "가나다".encode()) == [Section("가나다", None)]


def test_chunking_preserves_overlap_and_page():
    chunks = chunk_sections([Section("a" * 1100, 2)])
    assert len(chunks) == 2
    assert chunks[0].text == "a" * 1000
    assert chunks[1].text == "a" * 300
    assert chunks[1].page_number == 2


def test_rejects_empty_text():
    with pytest.raises(ValueError, match="텍스트"):
        extract_sections("empty.md", b"  ")


def test_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="형식"):
        extract_sections("x.docx", b"data")


def test_rejects_oversized_file():
    with pytest.raises(ValueError, match="10 MiB"):
        extract_sections("x.txt", b"a" * (10 * 1024 * 1024 + 1))


def test_rejects_broken_pdf():
    with pytest.raises(ValueError, match="PDF"):
        extract_sections("x.pdf", b"not a pdf")


def test_rejects_image_or_blank_pdf_without_text():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="텍스트"):
        extract_sections("blank.pdf", output.getvalue())


def test_rejects_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("demo")
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="암호화"):
        extract_sections("secret.pdf", output.getvalue())


def test_rejects_pdf_over_100_pages():
    writer = PdfWriter()
    for _ in range(101):
        writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="100페이지"):
        extract_sections("long.pdf", output.getvalue())


def test_embedding_dimension_checked_before_db_write():
    class FakeSession:
        def __init__(self):
            self.added = []

        def add(self, value):
            self.added.append(value)

    session = FakeSession()
    with pytest.raises(ValueError, match="1536"):
        ingest_document(session, "x.txt", b"hello", lambda texts: ([[0.1]], 1))
    assert session.added == []
