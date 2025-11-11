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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.groq_handler import generate_response, LANGUAGES, load_memory, save_memory, ensure_memory_file

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

UPLOAD_DIR = Path("/tmp/uploads")  
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Request models
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

# Routes
@app.get("/")
def home():
    ensure_memory_file()
    return {
        "status": "Aisha is ready!",
        "hint": "POST /chat or /chat/image"
    }

@app.get("/modes")
def modes():
    return {"mode": "aisha", "description": "Friendly best-friend style."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/memory")
def memory():
    return {"memory": load_memory()}

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
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message!")
    try:
        reply = generate_response(request.message, request.language)
        return {"reply": reply}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# FIXED IMAGE ENDPOINT
@app.post("/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    message: Optional[str] = None,
    language: str = "en"
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

    # TEXT FOR AI
    user_text = message.strip() if message and message.strip() else "Describe this image."

    try:
        
        reply = generate_response(
            user_message=user_text,
            language=language,
            image_path=str(file_path)   # FIXED!
        )

        # Memory update
        mem = load_memory()
        mem["conversations"].append({
            "role": "user",
            "msg": f"[Image: {filename}] {user_text}"
        })
        save_memory(mem)

        return {
            "reply": reply,
            "image_path": f"uploads/{filename}",   
            "filename": filename
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Vision error: {str(e)}")

# SERVE IMAGES FROM /tmp/uploads
app.mount("/uploads", StaticFiles(directory="/tmp/uploads"), name="uploads")