'use client'

import { useInitialization } from './InitializationProvider'

export default function InitializationStatus() {
  const { 
    isInitialized, 
    isInitializing, 
    initError, 
    backendStatus, 
    backendMode,
    ragStatus,
    retryInitialization 
  } = useInitialization()

  // Don't show anything when fully initialized (unless ultra-minimal mode)
  if (isInitialized && backendMode !== 'ultra_minimal') {
    return null
  }

  // Show minimal status for ultra-minimal mode
  if (isInitialized && backendMode === 'ultra_minimal') {
    return (
      <div className="fixed top-0 left-0 right-0 bg-green-600 text-white p-2 z-50 shadow-lg">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-lg">🪶</span>
            <span className="text-sm font-medium">Ultra-Minimal Mode Active - Lightweight AI Assistant Ready</span>
          </div>
          <div className="text-xs">Memory Optimized</div>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed top-0 left-0 right-0 bg-blue-600 text-white p-4 z-50 shadow-lg">
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {isInitializing ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span className="font-medium">
                {backendMode === 'ultra_minimal' ? '🪶 Starting Ultra-Minimal AI...' : '🚀 Initializing AI Tutoring System...'}
              </span>
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
              <span className="font-medium">
                {backendMode === 'ultra_minimal' 
                  ? 'Ultra-minimal system ready...' 
                  : 'System ready, initializing AI components...'}
              </span>
            </>
          )}
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm flex items-center space-x-1">
            <span>Mode:</span>
            <span className="text-xs bg-white/20 px-2 py-1 rounded">
              {backendMode === 'ultra_minimal' ? '🪶 Ultra-Minimal' :
               backendMode === 'minimal' ? '⚡ Minimal' :
               backendMode === 'full' ? '🔋 Full' : '❓ Unknown'}
            </span>
          </div>
          
          <div className="text-sm flex items-center space-x-1">
            <span>Backend:</span>
            <span className="text-lg">
              {backendStatus === 'healthy' ? '🟢' : 
               backendStatus === 'unhealthy' ? '🔴' : '🟡'}
            </span>
          </div>
          
          {ragStatus && (
            <div className="text-sm flex items-center space-x-1">
              <span>AI:</span>
              <span className="text-xs">
                {ragStatus === 'ready' ? '✅' :
                 ragStatus === 'loading' ? '⏳' :
                 ragStatus === 'error' ? '❌' : '❓'}
              </span>
            </div>
          )}
          
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