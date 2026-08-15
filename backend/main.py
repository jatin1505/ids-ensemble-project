
"""
FastAPI entry point.
 
Run from the project root (the folder containing both backend/ and
shared/):
 
    uvicorn backend.main:app --reload
 
Then visit:
    http://127.0.0.1:8000/health   -- plain health check
    http://127.0.0.1:8000/docs     -- FastAPI's auto-generated API docs,
                                       useful for poking at endpoints
                                       without writing a frontend first
"""
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from backend.websocket_manager import router as websocket_router
 
app = FastAPI(title="IDS Backend")
 
# The React dev server runs on a different port (usually 5173 or 3000)
# than FastAPI (8000). Browsers block requests between different origins
# by default -- this is CORS -- and without this middleware, Member B
# would just see a "Network Error" in the browser console with no
# useful detail about why. allow_origins=["*"] is fine for local
# development; narrow it to your actual deployed frontend URL later if
# you deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
app.include_router(websocket_router)
 
 
@app.get("/health")
def health_check():
    return {"status": "ok"}
 