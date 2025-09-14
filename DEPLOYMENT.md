# Deployment Guide

This guide helps you deploy:

- FastAPI backend to Render
- Next.js frontend to Vercel

## 1) Backend on Render

We added a `render.yaml` so you can use Render Blueprint Deploys.

### Steps

1. Push your repo to GitHub (done)
2. Go to Render → New + → Blueprint
3. Paste your repo URL; it will detect `render.yaml`
4. Set environment variables (Render → Service → Environment):
   - GEMINI_API_KEY (optional)
   - COHERE_API_KEY (optional)
   - PINECONE_API_KEY (optional)
   - PINECONE_INDEX_NAME (optional)
5. Deploy. Render will install Python 3.12, pip install requirements, and run `python run_backend.py`.

Notes:
- The service exposes port 8000; Render will map it to a public URL.
- Uploads will persist only while the service is running on the free plan; consider persistent storage for production.

## 2) Frontend on Vercel

The Next.js frontend lives under `frontend/`.

### Steps

1. In Vercel, import the GitHub repo
2. Set the project directory to `frontend`
3. Framework preset: Next.js
4. Environment Variables:
   - NEXT_PUBLIC_BACKEND_URL = https://<your-render-service>.onrender.com
5. Build command (default): `npm run build`
6. Output directory (default): `.next`
7. Deploy

### Local test against Render URL

```cmd
cd frontend
set NEXT_PUBLIC_BACKEND_URL=https://<your-render-service>.onrender.com
npm run dev
```

## 3) Health checks & verification

- Backend: curl https://<your-render-service>.onrender.com/health
- Frontend: open the Vercel URL, try Upload → Chat → Metrics → Evaluations

## 4) Troubleshooting

- 408/timeout on chat: increase timeouts are already set in `frontend/lib/api.ts` (120s for chat)
- 500 on metrics: ensure evaluator optional deps are available; metrics endpoint gracefully returns basic metrics if advanced evaluator missing
- Missing API keys: features using Gemini/Cohere/Pinecone may degrade; set keys in Render env

## 5) Optional improvements

- Add Vercel `vercel.json` for custom headers or rewrites
- Add Render disk for uploads if you need persistence across deploys
- Introduce logging/monitoring (e.g., Sentry) for both apps
