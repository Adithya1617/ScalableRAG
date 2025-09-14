# Independent Backend and Frontend Deployment

This guide shows how to deploy the **backend** and **frontend** as completely independent services.

## Architecture Overview

- **Backend**: FastAPI service deployed on Render
- **Frontend**: Next.js app deployed on Vercel
- **Communication**: Frontend connects to backend via environment variable (`NEXT_PUBLIC_BACKEND_URL`)

## 1. Deploy Backend to Render

### Option A: Blueprint Deployment (Recommended)

1. Push your repo to GitHub
2. Go to Render → New + → Blueprint
3. Paste your repo URL; it will detect `render.yaml`
4. Set environment variables in Render Dashboard:
   - `GEMINI_API_KEY` (your API key)
   - `COHERE_API_KEY` (your API key)
   - `PINECONE_API_KEY` (your API key)
   - `PINECONE_INDEX_NAME` (default: `quickstart`)
5. Deploy

### Option B: Manual Web Service

1. Go to Render → New + → Web Service
2. Connect your GitHub repo
3. Configure:
   - **Runtime**: Python 3.12
   - **Build Command**: `pip install -r requirements-render.txt`
   - **Start Command**: `cd rag_app && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
4. Set environment variables as above
5. Deploy

### Backend URL

After deployment, note your backend URL: `https://your-service-name.onrender.com`

## 2. Deploy Frontend to Vercel

### Steps

1. In Vercel, import the GitHub repo
2. Set the **Project Directory** to `frontend`
3. Framework preset: **Next.js**
4. **Environment Variables**:
   - `NEXT_PUBLIC_BACKEND_URL` = `https://your-render-service.onrender.com`
5. Build command (default): `npm run build`
6. Output directory (default): `.next`
7. Deploy

### Local Development

To test locally against the deployed backend:

```bash
cd frontend
npm install
export NEXT_PUBLIC_BACKEND_URL=https://your-render-service.onrender.com
npm run dev
```

## 3. Complete Independence

✅ **Backend** can be deployed and tested independently:
```bash
curl https://your-render-service.onrender.com/health
```

✅ **Frontend** can be deployed and tested independently:
- Open the Vercel URL
- It will automatically connect to the backend via `NEXT_PUBLIC_BACKEND_URL`

✅ **No shared code** - they communicate only via HTTP API

✅ **CORS configured** - backend allows requests from any origin

## 4. Environment Variables Summary

### Backend (Render)
| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Optional | For AI responses |
| `COHERE_API_KEY` | Optional | For document reranking |
| `PINECONE_API_KEY` | Optional | For vector search |
| `PINECONE_INDEX_NAME` | Optional | Index name (default: quickstart) |

### Frontend (Vercel)
| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | Yes | Backend service URL |

## 5. Testing Independence

Use the provided test scripts to verify your backend is ready for frontend integration:

### Windows
```cmd
test_backend_independence.bat https://your-backend.onrender.com
```

### Linux/Mac
```bash
chmod +x test_backend_independence.sh
./test_backend_independence.sh https://your-backend.onrender.com
```

The script tests:
- ✅ Health endpoint accessibility
- ✅ CORS configuration
- ✅ API endpoint availability

## 6. Troubleshooting

- **CORS Issues**: Backend has `allow_origins=["*"]` configured
- **Timeout Errors**: Frontend has 60s default, 120s for uploads/chat
- **Missing API Keys**: Features degrade gracefully without keys
- **Environment Variables**: Set in respective platform dashboards

The backend and frontend are now completely independent and can be deployed, updated, and scaled separately!
