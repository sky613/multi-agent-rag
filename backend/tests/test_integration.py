import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db import SessionLocal
from app.documents import delete_document, ingest_document
from app.models import Chunk
from app.retrieval import search_chunks
from app.main import app


@pytest.mark.skipif(os.getenv("RAG_INTEGRATION") != "1", reason="requires Compose PostgreSQL")
def test_document_vectors_are_searched_and_cascade_deleted():
    vector = [0.0] * 1536
    vector[0] = 1.0
    with SessionLocal() as session:
        document = ingest_document(session, f"integration-{uuid4()}.txt", b"revenue is ten",
                                   lambda texts: ([vector for _ in texts], 3))
        document_id = document.id
        chunk_id = document.chunks[0].id
        try:
            hits = search_chunks(session, vector, 1)
            assert hits[0].chunk_id == chunk_id
            assert hits[0].score == pytest.approx(1.0)
        finally:
            assert delete_document(session, document_id)
        assert session.scalar(select(func.count()).select_from(Chunk).where(Chunk.id == chunk_id)) == 0


@pytest.mark.skipif(os.getenv("RAG_INTEGRATION") != "1", reason="requires Compose PostgreSQL")
def test_upload_question_list_and_delete_api(monkeypatch):
    vector = [0.0] * 1536
    vector[0] = 1.0

    class FakeLLM:
        def embed(self, texts, usage=None):
            if usage is not None:
                usage.calls += 1
            return [vector for _ in texts], len(texts)

        def chat_json(self, system, user, usage):
            usage.calls += 1
            if "draft_answer" in system:
                import json
                chunk_id = json.loads(user)["evidence"][0]["chunk_id"]
                return {"draft_answer": "매출은 10이다", "claims": [{"text": "매출은 10이다", "chunk_ids": [chunk_id]}]}
            return {"verdicts": [{"claim_index": 0, "supported": True}]}

    monkeypatch.setattr("app.main.get_llm", lambda: FakeLLM())
    client = TestClient(app)
    filename = f"api-{uuid4()}.txt"
    uploaded = client.post("/api/documents", files={"file": (filename, b"revenue is ten")})
    assert uploaded.status_code == 201
    document_id = uploaded.json()["id"]
    try:
        assert any(d["id"] == document_id for d in client.get("/api/documents").json())
        answer = client.post("/api/questions", json={"question": "revenue?", "top_k": 1})
        assert answer.status_code == 200
        assert answer.json()["answer"] == "매출은 10이다"
        assert answer.json()["retrieved"][0]["document_id"] == document_id
        assert answer.json()["citations"][0]["document_id"] == document_id
    finally:
        assert client.delete(f"/api/documents/{document_id}").status_code == 204
