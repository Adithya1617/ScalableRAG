"use client"

import { uploadAndIndex } from '@/lib/api'
import React, { useState } from 'react'

export default function Upload({ onIndexed }: { onIndexed?: (files: string[]) => void }) {
  const [files, setFiles] = useState<FileList | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string>('')

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!files || files.length === 0) {
      setMessage('Please select at least one file.')
      return
    }
    setBusy(true)
    setMessage('Uploading and indexing...')
    try {
      const list = Array.from(files)
  const res = await uploadAndIndex(list)
  setMessage(res.message || 'Indexed successfully')
  if (onIndexed) onIndexed(res.files || [])
    } catch (err: any) {
      setMessage(err?.response?.data?.detail || err.message || 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card">
      <h2>📁 Upload & Index</h2>
      <p>Upload PDFs, TXT, DOCX or MD and build the index before chatting.</p>
      <form onSubmit={onSubmit} className="mt">
        <label className="label">Files</label>
        <input className="input" type="file" multiple accept=".pdf,.txt,.docx,.md" onChange={e => setFiles(e.target.files)} />
        <button className="btn primary mt" type="submit" disabled={busy}>{busy ? 'Processing...' : 'Upload & Build Index'}</button>
      </form>
      {message && <p className="mt badge">{message}</p>}
    </div>
  )
}
