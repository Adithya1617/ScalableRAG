'use client'

import { createContext, ReactNode, useContext, useEffect, useState } from 'react'
import { health, initializeRAG } from '../../lib/api'

interface InitializationState {
  isInitialized: boolean
  isInitializing: boolean
  initError: string | null
  backendStatus: 'unknown' | 'healthy' | 'unhealthy'
  backendMode: 'full' | 'minimal' | 'ultra_minimal' | 'unknown'
  ragStatus: string | null
  retryInitialization: () => void
}

const InitializationContext = createContext<InitializationState | null>(null)

export const useInitialization = () => {
  const context = useContext(InitializationContext)
  if (!context) {
    throw new Error('useInitialization must be used within InitializationProvider')
  }
  return context
}

interface InitializationProviderProps {
  children: ReactNode
}

export default function InitializationProvider({ children }: InitializationProviderProps) {
  const [isInitialized, setIsInitialized] = useState(false)
  const [isInitializing, setIsInitializing] = useState(true)
  const [initError, setInitError] = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<'unknown' | 'healthy' | 'unhealthy'>('unknown')
  const [backendMode, setBackendMode] = useState<'full' | 'minimal' | 'ultra_minimal' | 'unknown'>('unknown')
  const [ragStatus, setRagStatus] = useState<string | null>(null)

  const initializeSystem = async () => {
    setIsInitializing(true)
    setInitError(null)
    
    try {
      // First check if backend is healthy
      console.log('🔍 Checking backend health...')
      const healthData = await health()
      setBackendStatus('healthy')
      
      // Detect backend mode from health response
      if (healthData.mode === 'lightweight' || healthData.memory_efficient) {
        setBackendMode('ultra_minimal')
        console.log('🪶 Ultra-minimal backend detected')
        
        // Ultra-minimal mode is always ready
        setIsInitialized(true)
        setRagStatus('ready')
        setIsInitializing(false)
        return
      } else if (healthData.rag_status) {
        // New minimal backend with rag_status
        setBackendMode('minimal')
        setRagStatus(healthData.rag_status)
        
        if (healthData.rag_initialized || healthData.rag_status === 'ready') {
          console.log('✅ RAG already initialized')
          setIsInitialized(true)
          setIsInitializing(false)
          return
        }
      } else {
        // Legacy full backend
        setBackendMode('full')
        if (healthData.rag_initialized) {
          console.log('✅ RAG already initialized')
          setIsInitialized(true)
          setIsInitializing(false)
          return
        }
      }
      
      // Initialize RAG system if needed
      console.log('🚀 Initializing RAG system...')
      const initData = await initializeRAG()
      
      if (initData.status === 'ready' || initData.status === 'success') {
        console.log('✅ RAG system initialized successfully')
        setIsInitialized(true)
        setRagStatus(initData.status)
      } else {
        throw new Error(initData.message || 'RAG system failed to initialize properly')
      }
      
    } catch (error: any) {
      console.error('❌ Initialization failed:', error)
      setBackendStatus('unhealthy')
      
      // Extract error message
      let errorMessage = 'Unknown initialization error'
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message
      } else if (error.message) {
        errorMessage = error.message
      }
      
      setInitError(errorMessage)
    } finally {
      setIsInitializing(false)
    }
  }

  const retryInitialization = () => {
    initializeSystem()
  }

  useEffect(() => {
    initializeSystem()
  }, [])

  const value: InitializationState = {
    isInitialized,
    isInitializing,
    initError,
    backendStatus,
    backendMode,
    ragStatus,
    retryInitialization
  }

  return (
    <InitializationContext.Provider value={value}>
      {children}
    </InitializationContext.Provider>
  )
}