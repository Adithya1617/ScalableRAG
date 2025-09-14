"use client"

import { listEvaluations, runEvaluation } from '@/lib/api'
import { useEffect, useState } from 'react'

export default function Evaluations() {
  const [evaluations, setEvaluations] = useState<any[]>([])
  const [busy, setBusy] = useState(false)
  const [evalType, setEvalType] = useState('quick')

  const refresh = async () => {
    try {
      const res = await listEvaluations()
      setEvaluations(res.evaluations || [])
    } catch {}
  }

  useEffect(() => { refresh() }, [])

  const run = async () => {
    setBusy(true)
    try {
      await runEvaluation(evalType, { advanced_metrics: true })
      setTimeout(refresh, 2000)
    } catch {} finally { setBusy(false) }
  }

  return (
    <div className="card">
      <h2>📊 Evaluations</h2>
      <div className="row mt">
        <div className="col">
          <label className="label">Type</label>
          <select className="input" value={evalType} onChange={e => setEvalType(e.target.value)}>
            {['quick','comprehensive','performance','advanced'].map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div className="col">
          <button className="btn primary mt" onClick={run} disabled={busy}>{busy ? 'Running...' : 'Run Evaluation'}</button>
        </div>
      </div>

      <hr className="mt" />
      {evaluations.length === 0 && <p>No evaluations yet.</p>}
      {evaluations.map((e, idx) => (
        <div className="card" key={idx}>
          <strong>{e.timestamp || 'unknown'}</strong>
          <div className="row">
            <div className="col">Test Size: {e.test_dataset_size ?? '-'}</div>
            <div className="col">Avg Response: {typeof e.avg_response_time === 'number' ? `${e.avg_response_time.toFixed(2)}s` : '-'}</div>
            <div className="col">LLM Score: {typeof e.avg_llm_judge_score === 'number' ? e.avg_llm_judge_score.toFixed(1) : '-'}</div>
            <div className="col">ROUGE-1: {typeof e.rouge1 === 'number' ? `${(e.rouge1*100).toFixed(1)}%` : '-'}</div>
          </div>
        </div>
      ))}
    </div>
  )
}
