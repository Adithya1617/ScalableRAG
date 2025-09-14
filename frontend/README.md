# Frontend (Next.js) for RAG System

This is a minimal React (Next.js 14 + TypeScript) frontend for your FastAPI RAG backend.

## Features
- Upload and index documents (calls `/upload-and-index`)
- Chat with intelligent query (calls `/query/intelligent`)
- Optional real-time metrics per query (calls `/metrics/real-time/{query}`)
- List and run evaluations

## Prerequisites
- Node.js 18+

## Setup

```cmd
cd frontend
npm install
```

Optionally set the backend URL (defaults to http://localhost:8000):

```cmd
set NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## Run dev server

```cmd
npm run dev
```

Open http://localhost:3000

## Build and start (production)

```cmd
npm run build
npm run start
```

