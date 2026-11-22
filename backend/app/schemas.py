from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, StrictBool, field_validator


class QuestionInput(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("질문을 입력하세요")
        return value


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_number: int | None
    content: str
    score: float


class RetrievalInput(QuestionInput):
    pass


class RetrievalOutput(BaseModel):
    chunks: list[RetrievedChunk]


class Claim(BaseModel):
    text: str
    chunk_ids: list[UUID]


class AnalysisInput(BaseModel):
    question: str
    chunks: list[RetrievedChunk]


class AnalysisOutput(BaseModel):
    draft_answer: str
    claims: list[Claim]


class ValidationInput(AnalysisInput):
    draft_answer: str
    claims: list[Claim]


class Issue(BaseModel):
    claim: str
    reason: str


class ValidationOutput(BaseModel):
    supported: bool
    issues: list[Issue]
    final_answer: str
    citation_chunk_ids: list[UUID]


class Verdict(BaseModel):
    claim_index: int = Field(ge=0)
    supported: StrictBool
    reason: str = ""


class VerdictResponse(BaseModel):
    verdicts: list[Verdict]


class Usage(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    estimated_cost_usd: float | None = None
    cost_known: bool = Field(default=True, exclude=True)


class Citation(BaseModel):
    document_id: UUID
    filename: str
    chunk_id: UUID
    page_number: int | None
    excerpt: str
    score: float


class RetrievedSource(BaseModel):
    document_id: UUID
    filename: str
    chunk_id: UUID
    score: float


class QuestionOutput(BaseModel):
    answer: str
    retrieved: list[RetrievedSource]
    citations: list[Citation]
    validation: ValidationOutput
    usage: Usage
    timings_ms: dict[str, float]
    request_id: UUID


class DocumentOutput(BaseModel):
    id: UUID
    filename: str
    size_bytes: int
    chunk_count: int
    created_at: datetime
