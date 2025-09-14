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


## Deploy to Vercel

1. Import the GitHub repo in Vercel
2. Set Project Directory to `frontend`
3. Set Environment Variable:
	- `NEXT_PUBLIC_BACKEND_URL` = your Render backend URL (e.g., https://scalable-rag-backend.onrender.com)
4. Deploy

