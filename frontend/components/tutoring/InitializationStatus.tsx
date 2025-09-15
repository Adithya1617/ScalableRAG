'use client'

import { useInitialization } from './InitializationProvider'

export default function InitializationStatus() {
  const { 
    isInitialized, 
    isInitializing, 
    initError, 
    backendStatus, 
    retryInitialization 
  } = useInitialization()

  // Don't show anything when fully initialized
  if (isInitialized) {
    return null
  }

  return (
    <div className="fixed top-0 left-0 right-0 bg-blue-600 text-white p-4 z-50 shadow-lg">
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {isInitializing ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span className="font-medium">🚀 Initializing AI Tutoring System...</span>
            </>
          ) : initError ? (
            <>
              <span className="text-red-300 text-lg">❌</span>
              <div>
                <div className="font-medium">Initialization Failed</div>
                <div className="text-sm text-red-200 mt-1">{initError}</div>
              </div>
            </>
          ) : (
            <>
              <span className="text-yellow-300 text-lg">⏳</span>
              <span className="font-medium">System ready, initializing AI components...</span>
            </>
          )}
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm flex items-center space-x-1">
            <span>Backend:</span>
            <span className="text-lg">
              {backendStatus === 'healthy' ? '🟢' : 
               backendStatus === 'unhealthy' ? '🔴' : '🟡'}
            </span>
          </div>
          
          {initError && (
            <button
              onClick={retryInitialization}
              className="bg-white text-blue-600 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-100 transition-colors duration-200 shadow-sm"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  )
}