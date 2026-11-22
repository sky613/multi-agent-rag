export type Document = {
  id: string
  filename: string
  size_bytes: number
  chunk_count: number
  created_at: string
}

export type Answer = {
  answer: string
  citations: Array<{
    document_id: string
    filename: string
    chunk_id: string
    page_number: number | null
    excerpt: string
    score: number
  }>
  validation: { supported: boolean; issues: Array<{ claim: string; reason: string }> }
  usage: { calls: number; input_tokens: number; output_tokens: number; embedding_tokens: number; estimated_cost_usd: number | null }
  timings_ms: Record<string, number>
  request_id: string
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options)
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `요청 실패 (${response.status})`)
  }
  return response.status === 204 ? undefined as T : response.json()
}

export const api = {
  documents: () => request<Document[]>('/documents'),
  upload: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return request<Document>('/documents', { method: 'POST', body })
  },
  remove: (id: string) => request<void>(`/documents/${id}`, { method: 'DELETE' }),
  ask: (question: string, topK: number) => request<Answer>('/questions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  }),
}
