import axios from 'axios'

const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: BASE_URL,
  // Default timeout; individual calls can override
  timeout: 60000,
})

export const health = async () => (await api.get('/health')).data

export const uploadAndIndex = async (files: File[]) => {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const res = await api.post('/upload-and-index', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  })
  return res.data
}

export const intelligentQuery = async (payload: {
  query: string
  include_analysis?: boolean
  include_citations?: boolean
  metadata_filters?: Record<string, any> | null
}) => (await api.post('/query/intelligent', payload, { timeout: 120000 })).data

export const realtimeMetrics = async (query: string) => (
  await api.get(`/metrics/real-time/${encodeURIComponent(query)}`, { timeout: 60000 })
) .data

export const listEvaluations = async () => (await api.get('/evaluations')).data

export const runEvaluation = async (evaluation_type: string, options: Record<string, any> = {}) => (
  await api.post('/run-evaluation', { evaluation_type, options })
).data

export const evaluationStatus = async (evalId: string) => (
  await api.get(`/evaluation-status/${evalId}`)
).data

export const submitFeedback = async (payload: {
  query: string
  response: string
  rating: number
  feedback_text?: string
  session_id?: string
}) => (await api.post('/human-feedback', payload)).data
