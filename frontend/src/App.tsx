import { useEffect, useState } from 'react'
import { api, type Answer, type Document } from './api'

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [question, setQuestion] = useState('')
  const [topK, setTopK] = useState(5)
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.documents().then(setDocuments).catch((e) => setError(e.message))
  }, [])

  async function upload(file?: File) {
    if (!file) return
    setBusy(true); setError('')
    try {
      await api.upload(file)
      setDocuments(await api.documents())
    } catch (e) { setError((e as Error).message) }
    finally { setBusy(false) }
  }

  async function remove(id: string) {
    setBusy(true); setError('')
    try {
      await api.remove(id)
      setDocuments(await api.documents())
      setAnswer(null)
    } catch (e) { setError((e as Error).message) }
    finally { setBusy(false) }
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault()
    if (!question.trim()) return
    setBusy(true); setError(''); setAnswer(null)
    try { setAnswer(await api.ask(question, topK)) }
    catch (e) { setError((e as Error).message) }
    finally { setBusy(false) }
  }

  return <main className="shell">
    <header><span className="eyebrow">INTERNAL KNOWLEDGE</span><h1>Enterprise RAG</h1>
      <p>문서를 등록하고 근거가 있는 답변을 확인하세요.</p></header>
    {error && <div role="alert" className="error">{error}</div>}
    <div className="grid">
      <section className="card documents">
        <div className="section-heading"><h2>문서</h2><span>{documents.length}개</span></div>
        <label className="upload">PDF, TXT, Markdown 업로드
          <input type="file" accept=".pdf,.txt,.md" disabled={busy}
            onChange={(e) => { void upload(e.target.files?.[0]); e.target.value = '' }} />
        </label>
        <p className="hint">최대 10 MiB · PDF 100페이지</p>
        <ul className="document-list">{documents.map((doc) => <li key={doc.id}>
          <div><strong>{doc.filename}</strong><small>{doc.chunk_count} chunks · {(doc.size_bytes / 1024).toFixed(1)} KB</small></div>
          <button type="button" className="quiet" disabled={busy} onClick={() => void remove(doc.id)} aria-label={`${doc.filename} 삭제`}>삭제</button>
        </li>)}</ul>
        {documents.length === 0 && <p className="empty">등록된 문서가 없습니다.</p>}
      </section>
      <section className="card workspace">
        <div className="section-heading"><h2>질문하기</h2></div>
        <form onSubmit={(e) => void ask(e)}>
          <label htmlFor="question">문서에서 찾고 싶은 내용을 입력하세요</label>
          <textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={2000} rows={4} placeholder="예: 두 정책의 적용 대상을 비교해 주세요" />
          <div className="form-row"><label htmlFor="top-k">Top-K
            <input id="top-k" type="number" min={1} max={20} value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></label>
            <button type="submit" disabled={busy || !question.trim()}>{busy ? '처리 중…' : '질문 보내기'}</button></div>
        </form>
        {answer && <div className="result">
          <h3>답변</h3><p className="answer">{answer.answer}</p>
          {!answer.validation.supported && <p className="notice">일부 주장 또는 전체 답변의 근거가 부족합니다.</p>}
          {answer.validation.issues.length > 0 && <ul className="issues">{answer.validation.issues.map((issue, i) => <li key={i}>{issue.claim}: {issue.reason}</li>)}</ul>}
          <h3>출처</h3>
          {answer.citations.length ? <ol className="citations">{answer.citations.map((cite) => <li key={cite.chunk_id}>
            <strong>{cite.filename}{cite.page_number ? ` · ${cite.page_number}페이지` : ''}</strong>
            <span>유사도 {(cite.score * 100).toFixed(1)}%</span><p>{cite.excerpt}</p>
          </li>)}</ol> : <p className="empty">표시할 출처가 없습니다.</p>}
          <p className="metrics">API 호출 {answer.usage.calls}회 · 입력 {answer.usage.input_tokens} / 출력 {answer.usage.output_tokens} / Embedding {answer.usage.embedding_tokens} tokens
            {answer.usage.estimated_cost_usd !== null && ` · 예상 $${answer.usage.estimated_cost_usd.toFixed(6)}`}</p>
        </div>}
      </section>
    </div>
  </main>
}
