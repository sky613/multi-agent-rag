from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from collections.abc import Callable

from pypdf import PdfReader

from app.models import Chunk, Document


@dataclass(frozen=True)
class Section:
    text: str
    page_number: int | None


def extract_sections(filename: str, data: bytes) -> list[Section]:
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("파일 크기는 10 MiB를 초과할 수 없습니다")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise ValueError("지원하지 않는 파일 형식입니다")
    if suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("암호화 PDF는 지원하지 않습니다")
            if len(reader.pages) > 100:
                raise ValueError("PDF는 100페이지를 초과할 수 없습니다")
            sections = [Section(page.extract_text().strip(), index + 1)
                        for index, page in enumerate(reader.pages)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF를 읽을 수 없습니다") from exc
    else:
        try:
            sections = [Section(data.decode("utf-8").strip(), None)]
        except UnicodeDecodeError as exc:
            raise ValueError("UTF-8 텍스트만 지원합니다") from exc
    sections = [section for section in sections if section.text]
    if not sections:
        raise ValueError("추출된 텍스트가 없습니다")
    return sections


def chunk_sections(sections: list[Section]) -> list[Section]:
    chunks = []
    for section in sections:
        for start in range(0, len(section.text), 800):
            chunks.append(Section(section.text[start:start + 1000], section.page_number))
            if start + 1000 >= len(section.text):
                break
    return chunks


def ingest_document(session, filename: str, data: bytes,
                    embedder: Callable[[list[str]], tuple[list[list[float]], int]]) -> Document:
    sections = chunk_sections(extract_sections(filename, data))
    vectors, _ = embedder([section.text for section in sections])
    if len(vectors) != len(sections) or any(len(vector) != 1536 for vector in vectors):
        raise ValueError("Embedding은 각 Chunk마다 1536차원이어야 합니다")
    suffix = Path(filename).suffix.lower()
    media_type = {".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/markdown"}[suffix]
    document = Document(filename=Path(filename).name, media_type=media_type, size_bytes=len(data))
    document.chunks = [Chunk(chunk_index=i, page_number=section.page_number,
                             content=section.text, embedding=vector)
                       for i, (section, vector) in enumerate(zip(sections, vectors))]
    session.add(document)
    try:
        session.commit()
        session.refresh(document)
    except Exception:
        session.rollback()
        raise
    return document


def delete_document(session, document_id) -> bool:
    document = session.get(Document, document_id)
    if document is None:
        return False
    session.delete(document)
    session.commit()
    return True
