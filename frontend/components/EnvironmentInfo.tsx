'use client'

import { useEffect, useState } from 'react'

export default function EnvironmentInfo() {
  const [envInfo, setEnvInfo] = useState<{
    backendUrl: string
    nodeEnv: string
    vercelUrl?: string
  } | null>(null)

  useEffect(() => {
    setEnvInfo({
      backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || 'Not set',
      nodeEnv: process.env.NODE_ENV || 'unknown',
      vercelUrl: process.env.NEXT_PUBLIC_VERCEL_URL || 'Not deployed on Vercel'
    })
  }, [])

  // Only show in development mode
  if (process.env.NODE_ENV !== 'development' || !envInfo) {
    return null
  }

  return (
    <div className="fixed bottom-4 right-4 bg-gray-800 text-white p-3 rounded-lg text-xs max-w-sm z-50">
      <div className="font-bold mb-2">🔧 Environment Info</div>
      <div><strong>Backend URL:</strong> {envInfo.backendUrl}</div>
      <div><strong>Environment:</strong> {envInfo.nodeEnv}</div>
      {envInfo.vercelUrl !== 'Not deployed on Vercel' && (
        <div><strong>Vercel URL:</strong> {envInfo.vercelUrl}</div>
      )}
    </div>
  )
}