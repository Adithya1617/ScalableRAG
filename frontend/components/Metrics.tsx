"use client"

import { health, realtimeMetrics } from '@/lib/api'
import { useEffect, useState } from 'react'

export default function Metrics() {
  const [status, setStatus] = useState<any>(null)
  const [query, setQuery] = useState('')
  const [metrics, setMetrics] = useState<any>(null)

  useEffect(() => {
    (async () => {
      try { setStatus(await health()) } catch {}
    })()
  }, [])

  const fetchMetrics = async () => {
    if (!query.trim()) return
    try {
      const res = await realtimeMetrics(query)
      setMetrics(res)
    } catch {}
  }

  return (
    <div className="card">
      <h2>📈 Metrics</h2>
      <div className="mt">
        <p>Backend: <span className="badge">{status ? 'Healthy' : 'Unknown'}</span></p>
      </div>

      <div className="mt">
        <label className="label">Query (for real-time metrics)</label>
        <input className="input" value={query} onChange={e => setQuery(e.target.value)} placeholder="Type a query..." />
        <button className="btn mt" onClick={fetchMetrics}>Get Metrics</button>
      </div>

      {metrics && (
        <div className="card mt">
          <pre>{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}
