from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app


def test_question_rejects_blank_input():
    response = TestClient(app).post("/api/questions", json={"question": "   "})
    assert response.status_code == 422


def test_upload_rejects_unsupported_file():
    response = TestClient(app).post("/api/documents", files={"file": ("x.docx", b"data")})
    assert response.status_code == 400
    assert "형식" in response.json()["detail"]


def test_empty_database_answers_without_api_key(monkeypatch):
    class EmptySession:
        def scalar(self, query):
            return None

    app.dependency_overrides[get_session] = lambda: EmptySession()
    monkeypatch.setattr("app.main.get_llm", lambda: (_ for _ in ()).throw(AssertionError("LLM should not initialize")))
    try:
        response = TestClient(app).post("/api/questions", json={"question": "없는 문서?"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["retrieved"] == []
    assert "근거" in response.json()["answer"]
