export interface Document {
  id: string
  filename: string
  sections: number
  chunks: number
  status: 'ready'
  created_at: string
}

export interface Citation {
  id: string
  document_id: string
  filename: string
  page: number | null
  section: number
  text: string
  score: number
  label: number
}

export interface Answer {
  answer: string
  citations: Citation[]
  insufficient_evidence: boolean
}

export interface Health {
  status: string
  ollama: string
  missing_models: string[]
  embedding_model: string
  chat_model: string
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, init)
  } catch {
    throw new Error('Cannot reach the API. Check that the backend is running.')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' ? body.detail : `Request failed (${response.status}).`)
  }
  return response.status === 204 ? undefined as T : response.json()
}