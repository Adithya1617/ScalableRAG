import type { Metadata } from 'next'
import EnvironmentInfo from '../../components/EnvironmentInfo'
import InitializationProvider from '../../components/tutoring/InitializationProvider'
import InitializationStatus from '../../components/tutoring/InitializationStatus'
import './globals.css'

export const metadata: Metadata = {
  title: 'AI-Powered Intelligent Tutoring System',
  description: 'Personalized learning experiences powered by agentic RAG technology',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 min-h-screen">
        <InitializationProvider>
          <InitializationStatus />
          <div className="container">
            {children}
          </div>
          <EnvironmentInfo />
        </InitializationProvider>
      </body>
    </html>
  )
}
