'use client'

import { createContext, ReactNode, useContext, useEffect, useState } from 'react'
import { health, initializeRAG } from '../../lib/api'

interface InitializationState {
  isInitialized: boolean
  isInitializing: boolean
  initError: string | null
  backendStatus: 'unknown' | 'healthy' | 'unhealthy'
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

  const initializeSystem = async () => {
    setIsInitializing(true)
    setInitError(null)
    
    try {
      // First check if backend is healthy
      console.log('🔍 Checking backend health...')
      const healthData = await health()
      setBackendStatus('healthy')
      
      // Check if already initialized
      if (healthData.rag_initialized) {
        console.log('✅ RAG already initialized')
        setIsInitialized(true)
        setIsInitializing(false)
        return
      }
      
      // Initialize RAG system
      console.log('🚀 Initializing RAG system...')
      await initializeRAG()
      
      // Verify initialization
      const healthAfterInit = await health()
      if (healthAfterInit.rag_initialized) {
        console.log('✅ RAG system initialized successfully')
        setIsInitialized(true)
      } else {
        throw new Error('RAG system failed to initialize properly')
      }
      
    } catch (error: any) {
      console.error('❌ Initialization failed:', error)
      setBackendStatus('unhealthy')
      
      // Extract error message
      let errorMessage = 'Unknown initialization error'
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
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
    retryInitialization
  }

  return (
    <InitializationContext.Provider value={value}>
      {children}
    </InitializationContext.Provider>
  )
}