"use client"

import { intelligentQuery, realtimeMetrics } from '@/lib/api'
import { useState } from 'react'
import { useInitialization } from './tutoring/InitializationProvider'

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
  
  const { isInitialized, isInitializing, initError, backendMode } = useInitialization()

  const send = async () => {
    if (!text.trim() || busy || !isInitialized) return
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
      
      // Handle different response formats based on backend mode
      const bot: Message = {
        role: 'bot',
        content: res.response || 'No response',
        query_analysis: res.analysis || res.query_analysis, // Ultra-minimal uses 'analysis'
        citations: res.citations || res.sources || [] // Handle both formats
      }
      
      // Add backend mode info for ultra-minimal responses
      if (backendMode === 'ultra_minimal' && res.rag_status) {
        bot.content += `\n\n_Note: Response from ${backendMode} mode (${res.rag_status})_`
      }
      
      // Show the main response immediately
      setMessages(m => [...m, bot])

      // Fetch advanced metrics only if backend supports it (not in ultra-minimal mode)
      if (enableAdvanced && backendMode !== 'ultra_minimal') {
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

  const isDisabled = !isInitialized || isInitializing || initError !== null

  return (
    <div className="card">
      <h2>💬 Chat</h2>
      
      {/* Backend Mode Status */}
      {backendMode === 'ultra_minimal' && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <span className="text-lg">🪶</span>
            <div>
              <div className="text-sm font-medium text-blue-800">Ultra-Minimal Mode Active</div>
              <div className="text-xs text-blue-600">Lightweight AI assistant with built-in knowledge base. Full features available with upgraded backend.</div>
            </div>
          </div>
        </div>
      )}
      
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
        {initError ? (
          <div className="card" style={{borderColor: '#ef4444', backgroundColor: '#fef2f2'}}>
            <p style={{color: '#dc2626'}}>❌ System initialization failed. Please refresh the page to retry.</p>
          </div>
        ) : !isInitialized ? (
          <div className="card" style={{borderColor: '#f59e0b', backgroundColor: '#fffbeb'}}>
            <p style={{color: '#d97706'}}>⏳ Initializing AI system, please wait...</p>
          </div>
        ) : null}
        
        <input 
          className="input" 
          placeholder={
            isDisabled 
              ? "Please wait for system initialization..." 
              : "Ask a question..."
          } 
          value={text} 
          onChange={e => setText(e.target.value)} 
          onKeyDown={e => { if (e.key === 'Enter') send() }}
          disabled={isDisabled}
        />
        <button 
          className="btn primary mt" 
          onClick={send} 
          disabled={isDisabled || busy || !text.trim()}
        >
          {busy ? 'Sending...' : 'Send'}
        </button>
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
