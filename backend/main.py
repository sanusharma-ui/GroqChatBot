import sys
import os
import traceback
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
import shutil
from pathlib import Path
import uuid
import mimetypes
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.groq_handler import (
    generate_response,
    PERSONAS,
    ensure_persona_memory,
    load_persona_memory,
    save_persona_memory
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Aisha — Friendly AI",
    description="Aisha: your warm, caring AI best friend. Uses Groq under the hood and persona-based modes.",
    version="2.0"
)

# --- CORS CONFIG ---
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

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- MODELS ---
class ChatRequest(BaseModel):
    message: str
    language: str = "en"

class ImageChatRequest(BaseModel):
    message: Optional[str] = None
    language: str = "en"

class UpdateUserMeta(BaseModel):
    name: Optional[str] = None
    interests: Optional[List[str]] = None
    notes: Optional[Dict[str, str]] = None

# --- ROUTES ---
@app.get("/")
def home():
    ensure_persona_memory("default")
    return {
        "status": "Aisha is ready!",
        "hint": "POST /chat or /chat/image",
        "available_modes": list(PERSONAS.keys())
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/memory")
def memory():
    return {"memory": load_persona_memory("default")}

@app.post("/memory/update")
def memory_update(payload: UpdateUserMeta):
    mem = load_persona_memory("default")
    if payload.name:
        mem["user"]["name"] = payload.name
    if payload.interests:
        mem["user"]["interests"] = payload.interests
    if payload.notes:
        mem["user"]["notes"].update(payload.notes)
    save_persona_memory("default", mem)
    return {"status": "ok"}

@app.get("/modes/list")
def list_modes():
    return {"modes": {k: v["name"] for k, v in PERSONAS.items()}}

# --- CHAT ROUTE (supports mode and reset) ---
@app.post("/chat")
def chat(request: ChatRequest, mode: str = "default", reset: bool = False):
    if mode not in PERSONAS:
        mode = "default"
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message!")
    try:
        # Reset memory if requested
        if reset:
            logger.info(f"Resetting memory for persona: {mode}")
            mem = {"user": {"name": None, "interests": [], "notes": {}}, "conversations": []}
            save_persona_memory(mode, mem)
        logger.info(f"Loading memory for persona: {mode}")
        reply = generate_response(
            user_message=request.message,
            persona_key=mode,
            language=request.language
        )
        # Per-persona memory
        mem = load_persona_memory(mode)
        mem["conversations"].append({
            "role": "user",
            "msg": request.message[:200]
        })
        mem["conversations"].append({
            "role": "assistant",
            "msg": reply[:200]
        })
        if len(mem["conversations"]) > 60:
            mem["conversations"] = mem["conversations"][-60:]
        save_persona_memory(mode, mem)
        return {
            "reply": reply,
            "mode": mode,
            "display_name": PERSONAS[mode]["name"]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# --- IMAGE CHAT ROUTE (supports mode) ---
@app.post("/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    message: Optional[str] = None,
    language: str = "en",
    mode: str = "default"
):
    # Validate type
    allowed = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, GIF, WebP allowed!")
    # Validate size (5MB)
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too big! Max 5MB.")
    # Save file
    ext = mimetypes.guess_extension(file.content_type) or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(content)
    # User text
    user_text = message.strip() if message and message.strip() else "Describe this image."
    try:
        reply = generate_response(
            user_message=user_text,
            persona_key=mode,
            language=language,
            image_path=str(file_path)
        )
        # Per-persona memory save
        mem = load_persona_memory(mode)
        mem["conversations"].append({
            "role": "user",
            "msg": f"[Image: {filename}] {user_text}"[:200]
        })
        mem["conversations"].append({
            "role": "assistant",
            "msg": reply[:200]
        })
        if len(mem["conversations"]) > 60:
            mem["conversations"] = mem["conversations"][-60:]
        save_persona_memory(mode, mem)
        return {
            "reply": reply,
            "image_path": f"uploads/{filename}",
            "filename": filename,
            "mode": mode,
            "display_name": PERSONAS.get(mode, PERSONAS["default"])["name"]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Vision error: {str(e)}")

# --- SERVE IMAGES STATICALLY ---
app.mount("/uploads", StaticFiles(directory="/tmp/uploads"), name="uploads") 