from sqlalchemy import select

from app.models import Chunk, Document
from app.schemas import RetrievedChunk


def search_chunks(session, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
    if len(embedding) != 1536:
        raise ValueError("질문 Embedding은 1536차원이어야 합니다")
    distance = Chunk.embedding.cosine_distance(embedding).label("distance")
    rows = session.execute(
        select(Chunk, Document.filename, distance)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    ).all()
    return [RetrievedChunk(chunk_id=chunk.id, document_id=chunk.document_id,
                           filename=filename, page_number=chunk.page_number,
                           content=chunk.content, score=1 - float(dist))
            for chunk, filename, dist in rows]
