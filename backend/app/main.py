import logging
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAIError
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.agents import AnalysisAgent, Orchestrator, RetrievalAgent, ValidationAgent
from app.db import get_session, init_db
from app.documents import delete_document, ingest_document
from app.llm import LLMClient
from app.models import Chunk, Document
from app.schemas import DocumentOutput, QuestionInput, QuestionOutput, Usage


logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Enterprise RAG MVP", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:15173", "http://127.0.0.1:15173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(OpenAIError)
async def llm_error(request, exc):
    logging.error("external_api_error type=%s", type(exc).__name__)
    return JSONResponse(status_code=502, content={"detail": "외부 AI API 호출에 실패했습니다"})


@app.exception_handler(SQLAlchemyError)
async def db_error(request, exc):
    logging.error("database_error type=%s", type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": "데이터베이스를 사용할 수 없습니다"})


def get_llm() -> LLMClient:
    try:
        return LLMClient()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/health")
def health(session: Session = Depends(get_session)):
    session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/api/documents", response_model=DocumentOutput, status_code=201)
def upload_document(file: UploadFile = File(...), session: Session = Depends(get_session)):
    data = file.file.read(10 * 1024 * 1024 + 1)
    usage = Usage()
    try:
        document = ingest_document(session, file.filename or "", data,
                                   lambda texts: get_llm().embed(texts, usage))
    except ValueError as exc:
        status = 413 if len(data) > 10 * 1024 * 1024 else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    finally:
        logging.info("ingestion calls=%s embedding_tokens=%s", usage.calls, usage.embedding_tokens)
    return DocumentOutput(id=document.id, filename=document.filename, size_bytes=document.size_bytes,
                          chunk_count=len(document.chunks), created_at=document.created_at)


@app.get("/api/documents", response_model=list[DocumentOutput])
def list_documents(session: Session = Depends(get_session)):
    documents = session.execute(select(Document, func.count(Chunk.id))
                                .outerjoin(Chunk, Chunk.document_id == Document.id)
                                .group_by(Document.id).order_by(Document.created_at.desc())).all()
    return [DocumentOutput(id=d.id, filename=d.filename, size_bytes=d.size_bytes,
                           chunk_count=count, created_at=d.created_at) for d, count in documents]


@app.delete("/api/documents/{document_id}", status_code=204)
def remove_document(document_id: UUID, session: Session = Depends(get_session)):
    if not delete_document(session, document_id):
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다")
    return Response(status_code=204)


@app.post("/api/questions", response_model=QuestionOutput)
def ask(request: QuestionInput, session: Session = Depends(get_session)):
    llm = get_llm() if session.scalar(select(Chunk.id).limit(1)) is not None else None
    orchestrator = Orchestrator(RetrievalAgent(session, llm), AnalysisAgent(llm), ValidationAgent(llm))
    try:
        return orchestrator.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
