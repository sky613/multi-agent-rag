import json
import logging
from uuid import uuid4

from app.agents import Orchestrator, ValidationAgent
from app.schemas import (
    Claim, QuestionInput, RetrievalOutput, RetrievedChunk,
    Usage, ValidationInput,
)


def test_no_hits_skips_llm_and_reports_insufficient_evidence(caplog):
    class Retrieval:
        def run(self, request, usage):
            return RetrievalOutput(chunks=[])

    class MustNotRun:
        def run(self, request, usage):
            raise AssertionError("LLM should not run")

    with caplog.at_level(logging.INFO):
        result = Orchestrator(Retrieval(), MustNotRun(), MustNotRun()).run(
            QuestionInput(question="없는 내용", top_k=5))
    assert "근거" in result.answer
    assert result.citations == []
    assert result.usage.calls == 0
    assert '"chunk_ids": []' in caplog.text
    assert '"agent": "orchestrator"' in caplog.text


def test_validation_removes_unsupported_claim_and_fake_citation():
    good_id, fake_id = uuid4(), uuid4()
    chunk = RetrievedChunk(chunk_id=good_id, document_id=uuid4(), filename="a.txt",
                           page_number=None, content="매출은 10이다", score=0.9)

    class Chat:
        def chat_json(self, system, user, usage):
            return {"verdicts": [{"claim_index": 0, "supported": True},
                                 {"claim_index": 1, "supported": True}]}

    request = ValidationInput(question="매출은?", chunks=[chunk], draft_answer="매출은 10이고 이익은 20이다",
                              claims=[Claim(text="매출은 10이다", chunk_ids=[good_id]),
                                      Claim(text="이익은 20이다", chunk_ids=[fake_id])])
    result = ValidationAgent(Chat()).run(request, Usage())
    assert result.final_answer == "매출은 10이다"
    assert result.citation_chunk_ids == [good_id]
    assert result.supported is False
    assert len(result.issues) == 1


def test_validation_sends_only_cited_evidence():
    cited, other = uuid4(), uuid4()
    chunks = [RetrievedChunk(chunk_id=identifier, document_id=uuid4(), filename="x.txt",
                             page_number=None, content=content, score=0.8)
              for identifier, content in [(cited, "휴가는 15일"), (other, "경비는 20원")]]

    class Chat:
        def chat_json(self, system, user, usage):
            payload = json.loads(user)
            assert payload["claims"][0]["evidence"] == [{"chunk_id": str(cited), "text": "휴가는 15일"}]
            return {"verdicts": [{"claim_index": 0, "supported": True}]}

    result = ValidationAgent(Chat()).run(ValidationInput(
        question="휴가는?", chunks=chunks, draft_answer="휴가는 15일", claims=[Claim(text="휴가는 15일", chunk_ids=[cited])]), Usage())
    assert result.supported


def test_validation_rejects_claim_when_model_finds_no_support():
    chunk_id = uuid4()
    chunk = RetrievedChunk(chunk_id=chunk_id, document_id=uuid4(), filename="x.txt",
                           page_number=None, content="휴가는 15일", score=0.8)

    class Chat:
        def chat_json(self, system, user, usage):
            return {"verdicts": [{"claim_index": 0, "supported": False, "reason": "없음"}]}

    result = ValidationAgent(Chat()).run(ValidationInput(
        question="수당은?", chunks=[chunk], draft_answer="수당 20원",
        claims=[Claim(text="수당 20원", chunk_ids=[chunk_id])]), Usage())
    assert result.citation_chunk_ids == []
    assert "근거" in result.final_answer
    assert not result.supported
