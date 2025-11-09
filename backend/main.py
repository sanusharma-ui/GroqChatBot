
import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict

from backend.groq_handler import generate_response, LANGUAGES, load_memory, save_memory, update_memory_after_conversation, ensure_memory_file

app = FastAPI(
    title="Aisha — Friendly AI",
    description="Aisha: your warm, caring AI best friend. Uses Groq under the hood and a small memory layer.",
    version="1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
        "https://sonu-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str
    language: str = "en"

# Basic routes
@app.get("/")
def home():
    ensure_memory_file()
    return {
        "status": "Aisha is ready! 💖",
        "hint": "POST /chat with JSON { message, language }"
    }

@app.get("/modes")
def modes():
    return {"mode": "aisha", "description": "Friendly best-friend style (Hinglish-friendly)."}

@app.get("/health")
def health():
    return {"status": "ok", "detail": "Aisha backend alive"}

# Read memory (dev only)
@app.get("/memory")
def memory():
    mem = load_memory()
    return {"memory": mem}

# Endpoint to update user metadata (optional)
class UpdateUserMeta(BaseModel):
    name: Optional[str] = None
    interests: Optional[List[str]] = None
    notes: Optional[Dict[str,str]] = None

@app.post("/memory/update")
def memory_update(payload: UpdateUserMeta):
    mem = load_memory()
    if payload.name:
        mem["user"]["name"] = payload.name
    if payload.interests:
        mem["user"]["interests"] = payload.interests
    if payload.notes:
        mem["user"]["notes"].update(payload.notes)
    save_memory(mem)
    return {"status": "ok", "message": "Memory updated"}

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Khaali message mat bhej yaar 😄")

    try:
        reply = generate_response(request.message, request.language)
        return {"reply": reply}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
