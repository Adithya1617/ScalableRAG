# Frontend (Next.js) for RAG System

This is a **completely independent** React (Next.js 14 + TypeScript) frontend that communicates with your FastAPI RAG backend via HTTP API only.

## Architecture

- **No shared code** with backend
- **Environment-based configuration** - backend URL set via `NEXT_PUBLIC_BACKEND_URL`
- **CORS-enabled** - backend allows cross-origin requests
- **Deployable anywhere** - Vercel, Netlify, or any static hosting

## Features

- 📁 Upload and index documents (calls `/upload-and-index`)
- 💬 Chat with intelligent query (calls `/query/intelligent`)
- � List and run evaluations
- 📈 Advanced metrics (optional): fetched in the background after a chat reply and shown inline under the message if available.

## Prerequisites

- Node.js 18+

## Setup

```bash
cd frontend
npm install
```

## Configuration

Set the backend URL via environment variable:

```bash
# For local backend
export NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# For deployed backend
export NEXT_PUBLIC_BACKEND_URL=https://your-backend.onrender.com
```

**Default**: `http://localhost:8000` if not set

## Development

```bash
npm run dev
```

Open http://localhost:3000

## Production Build

```bash
npm run build
npm run start
```

## Deploy to Vercel

1. Import the GitHub repo in Vercel
2. Set **Project Directory** to `frontend`
3. Set **Environment Variable**:
   - `NEXT_PUBLIC_BACKEND_URL` = `https://your-render-backend.onrender.com`
4. Deploy

## Deploy to Other Platforms

Works with any static hosting:

- **Netlify**: Set `NEXT_PUBLIC_BACKEND_URL` in build settings
- **GitHub Pages**: Use `next export` and set env vars
- **AWS S3 + CloudFront**: Build and upload static files

## API Integration

The frontend calls these backend endpoints:

- `GET /health` - Health check
- `POST /upload-and-index` - Upload files
- `POST /query/intelligent` - Chat with AI
- `GET /metrics/real-time/{query}` - Real-time metrics (used by chat to show advanced metrics inline)
- `GET /evaluations` - List evaluations
- `POST /run-evaluation` - Run evaluation
- `GET /evaluation-status/{id}` - Check evaluation status

## Complete Independence

✅ **Backend URL** configured via environment variable
✅ **No code sharing** between frontend and backend
✅ **CORS configured** on backend for cross-origin requests
✅ **Deployable separately** with different CI/CD pipelines
✅ **Scalable independently** - backend and frontend can scale separately

