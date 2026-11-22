import pytest

from app.agents import AnalysisAgent, ValidationAgent
from app.schemas import AnalysisInput, Claim, RetrievedChunk, Usage, ValidationInput
from uuid import uuid4


@pytest.mark.parametrize("bad", [{}, {"verdicts": None}])
def test_malformed_validation_response_retries_then_fails_cleanly(bad):
    chunk_id = uuid4()
    chunk = RetrievedChunk(chunk_id=chunk_id, document_id=uuid4(), filename="x.txt",
                           page_number=None, content="휴가 15일", score=0.9)

    class Chat:
        calls = 0

        def chat_json(self, system, user, usage):
            self.calls += 1
            return bad

    client = Chat()
    with pytest.raises(ValueError, match="검증 결과 형식"):
        ValidationAgent(client).run(ValidationInput(question="휴가?", chunks=[chunk], draft_answer="휴가 15일",
                                                   claims=[Claim(text="휴가 15일", chunk_ids=[chunk_id])]), Usage())
    assert client.calls == 2


def test_malformed_analysis_response_fails_without_echoing_model_output():
    chunk = RetrievedChunk(chunk_id=uuid4(), document_id=uuid4(), filename="x.txt",
                           page_number=None, content="private document", score=0.9)

    class Chat:
        calls = 0

        def chat_json(self, system, user, usage):
            self.calls += 1
            return {"draft_answer": "private document", "claims": "bad"}

    client = Chat()
    with pytest.raises(ValueError, match="분석 결과 형식") as error:
        AnalysisAgent(client).run(AnalysisInput(question="?", chunks=[chunk]), Usage())
    assert "private document" not in str(error.value)
    assert client.calls == 2
