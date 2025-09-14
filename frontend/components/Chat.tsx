"use client"

import { intelligentQuery, realtimeMetrics } from '@/lib/api'
import { useState } from 'react'

type Message = {
  role: 'user' | 'bot'
  content: string
  query_analysis?: any
  citations?: any[]
  advanced_metrics?: any
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [includeAnalysis, setIncludeAnalysis] = useState(true)
  const [includeCitations, setIncludeCitations] = useState(true)
  const [enableAdvanced, setEnableAdvanced] = useState(true)
  const [categoryFilter, setCategoryFilter] = useState('All Categories')

  const send = async () => {
    if (!text.trim() || busy) return
    setBusy(true)
    const question = text
    setMessages(m => [...m, { role: 'user', content: question }])
    setText('')

    try {
      const payload: any = {
        query: question,
        include_analysis: includeAnalysis,
        include_citations: includeCitations,
        metadata_filters: categoryFilter === 'All Categories' ? null : { document_category: categoryFilter.toLowerCase() }
      }
      const res = await intelligentQuery(payload)
      const bot: Message = {
        role: 'bot',
        content: res.response || 'No response',
        query_analysis: res.query_analysis,
        citations: res.citations || []
      }
      // Show the main response immediately
      setMessages(m => [...m, bot])

      // Fetch advanced metrics in the background and append a separate message when ready
      if (enableAdvanced) {
        ;(async () => {
          try {
            const metrics = await realtimeMetrics(question)
            if (metrics && metrics.real_time_metrics) {
              const metricsMsg: Message = {
                role: 'bot',
                content: '(advanced metrics)',
                advanced_metrics: metrics.real_time_metrics
              }
              setMessages(m => [...m, metricsMsg])
            }
          } catch {
            // ignore metrics errors; keep main response
          }
        })()
      }
    } catch (err: any) {
      setMessages(m => [...m, { role: 'bot', content: err?.response?.data?.detail || err.message || 'Error' }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>💬 Chat</h2>
      <div className="row mt">
        <div className="col">
          <label className="label">Category Filter</label>
          <select className="input" value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
            {['All Categories','technical','business','academic','legal','medical','general'].map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>
        <div className="col">
          <label className="label">Options</label>
          <div>
            <label><input type="checkbox" checked={includeAnalysis} onChange={e => setIncludeAnalysis(e.target.checked)} /> Query Analysis</label>
            <span className="badge" style={{marginLeft:8}}>
              <label><input type="checkbox" checked={includeCitations} onChange={e => setIncludeCitations(e.target.checked)} /> Citations</label>
            </span>
            <span className="badge" style={{marginLeft:8}}>
              <label><input type="checkbox" checked={enableAdvanced} onChange={e => setEnableAdvanced(e.target.checked)} /> Advanced Metrics</label>
            </span>
          </div>
        </div>
      </div>

      <div className="mt">
        <input className="input" placeholder="Ask a question..." value={text} onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') send() }} />
        <button className="btn primary mt" onClick={send} disabled={busy}>{busy ? 'Sending...' : 'Send'}</button>
      </div>

      <hr className="mt" />

      <div className="mt">
        {messages.map((m, i) => (
          <div key={i} className="card" style={{borderColor: m.role === 'user' ? '#2563eb' : '#374151'}}>
            <strong>{m.role === 'user' ? 'You' : 'Assistant'}</strong>
            <p>{m.content}</p>

            {m.query_analysis && (
              <details>
                <summary>🧠 Query Analysis</summary>
                <pre>{JSON.stringify(m.query_analysis, null, 2)}</pre>
              </details>
            )}

            {Array.isArray(m.citations) && m.citations.length > 0 && (
              <details>
                <summary>📖 Citations ({m.citations.length})</summary>
                <ul>
                  {m.citations.slice(0, 5).map((c, idx) => (
                    <li key={idx}>{c.filename || 'Unknown'} (conf: {typeof c.confidence_score === 'number' ? c.confidence_score.toFixed(2) : c.confidence_score})</li>
                  ))}
                </ul>
              </details>
            )}

            {m.advanced_metrics && (
              <details>
                <summary>📈 Advanced Metrics</summary>
                <pre>{JSON.stringify(m.advanced_metrics, null, 2)}</pre>
              </details>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
