"use client"

import { uploadAndIndex } from '@/lib/api'
import React, { useState } from 'react'
import { useInitialization } from './tutoring/InitializationProvider'

export default function Upload({ onIndexed }: { onIndexed?: (files: string[]) => void }) {
  const { backendMode } = useInitialization()
  const [files, setFiles] = useState<FileList | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string>('')

  const isUltraMinimal = backendMode === 'ultra_minimal'

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!files || files.length === 0) {
      setMessage('Please select at least one file.')
      return
    }
    
    // Handle ultra-minimal mode
    if (isUltraMinimal) {
      setMessage('Upload not available in ultra-minimal mode. The system uses a built-in knowledge base.')
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
      
      {isUltraMinimal ? (
        <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-lg">🪶</span>
            <div className="text-sm font-medium text-amber-800">Ultra-Minimal Mode</div>
          </div>
          <p className="text-sm text-amber-700">
            File uploads are not available in ultra-minimal mode. The system uses a built-in knowledge base 
            for AI responses. Upgrade to the full backend for document indexing capabilities.
          </p>
        </div>
      ) : (
        <>
          <p>Upload PDFs, TXT, DOCX or MD and build the index before chatting.</p>
          <form onSubmit={onSubmit} className="mt">
            <label className="label">Files</label>
            <input 
              className="input" 
              type="file" 
              multiple 
              accept=".pdf,.txt,.docx,.md" 
              onChange={e => setFiles(e.target.files)}
              disabled={isUltraMinimal} 
            />
            <button 
              className="btn primary mt" 
              type="submit" 
              disabled={busy || isUltraMinimal}
            >
              {busy ? 'Processing...' : 'Upload & Build Index'}
            </button>
          </form>
          {message && <p className="mt badge">{message}</p>}
        </>
      )}
    </div>
  )
}
