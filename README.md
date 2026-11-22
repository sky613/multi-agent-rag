# Enterprise Multi-Agent RAG MVP

PDF, TXT, Markdown 문서를 PostgreSQL/pgvector에 등록하고, 검색 근거를 바탕으로 답하는 로컬/내부 데모입니다. Agent는 Retrieval → Analysis → Validation 순서로 실행됩니다. 앱과 DB는 Docker Compose에서 실행되며 문서 Chunk와 질문은 설정된 외부 OpenAI-compatible API로 전송됩니다.

## 시작하기

1. `.env.example`을 `.env`로 복사합니다. `POSTGRES_PASSWORD`에 로컬 DB 암호를 설정하고, `OPENAI_API_KEY`를 입력합니다. 필요한 경우 `OPENAI_BASE_URL`, `CHAT_MODEL`, `EMBEDDING_MODEL`을 변경합니다. `.env`는 Git에서 제외됩니다.
2. `docker compose up --build -d`를 실행합니다.
3. UI: <http://localhost:15173>, API: <http://localhost:18000/docs>, Health: <http://localhost:18000/api/health>.

`EMBEDDING_MODEL`은 **1536차원** 벡터를 반환해야 합니다. 기본값은 `text-embedding-3-small`입니다. 모델 차원을 바꾸려면 DB 스키마와 기존 벡터를 재생성해야 합니다. 단가를 알고 있으면 `.env`의 `CHAT_INPUT_USD_PER_MILLION`, `CHAT_OUTPUT_USD_PER_MILLION`, `EMBEDDING_USD_PER_MILLION`을 모두 설정해 예상 USD 비용을 표시할 수 있습니다. 실제 사용한 항목의 단가가 빠지면 비용은 `null`입니다.

API는 `POST /api/documents`(multipart `file`), `GET /api/documents`, `DELETE /api/documents/{id}`, `POST /api/questions`(`{"question":"...","top_k":5}`)를 제공합니다. 질문 응답의 `retrieved`는 검색 Top-K이고 `citations`는 검증된 답변에 실제 사용한 출처입니다.

## 검증

```bash
docker compose config --quiet
docker compose exec -T backend pytest -q
docker compose exec -T -e RAG_INTEGRATION=1 backend pytest -q tests/test_integration.py
docker compose exec -T backend python -m eval.evaluate --base-url http://localhost:8000
```

마지막 평가 명령은 유효한 외부 API key가 필요합니다. Synthetic 문서 2개를 임시 등록해 retrieval Recall@1, 답변할 수 없는 질문의 보류율, 질문 응답 시간 p50/p95, 호출·token·예상 비용을 계산한 뒤 문서를 삭제합니다. 별도 Agent 테스트는 근거 없는 주장을 제거하는지 확인합니다. 이 수치는 실제 기업 문서의 검색 품질을 보증하지 않습니다.

## 범위와 제한

- 인증·조직 분리 없이 단일 조직의 내부 데모를 가정합니다. Compose 포트는 `127.0.0.1`에만 바인딩하며 외부에 공개하면 안 됩니다.
- 최대 10 MiB, PDF 최대 100페이지입니다. 이미지 PDF의 OCR은 제공하지 않습니다.
- Embedding/Chat API 오류는 해당 요청의 오류로 반환합니다. AI 응답의 의미 검증에는 모델 판정이 포함되므로 근거 오류를 완전히 배제할 수는 없습니다.
- 구조·DB/API·Agent 상세 설계는 [설계 문서](docs/superpowers/specs/2026-11-18-enterprise-rag-design.md)를 참고하세요.
# multi-agent-rag
