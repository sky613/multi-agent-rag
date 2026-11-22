from app.models import Chunk, Document


def test_chunk_schema_cascades_and_uses_1536_vector():
    fk = next(iter(Chunk.__table__.c.document_id.foreign_keys))
    assert fk.ondelete == "CASCADE"
    assert Chunk.__table__.c.embedding.type.dim == 1536
    assert any({"document_id", "chunk_index"} == {c.name for c in constraint.columns}
               for constraint in Chunk.__table__.constraints if hasattr(constraint, "columns"))
    assert Document.__tablename__ == "documents"
