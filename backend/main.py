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

# Setup uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Request model
class ChatRequest(BaseModel):
    message: str
    language: str = "en"

class ImageChatRequest(BaseModel):
    message: Optional[str] = None
    language: str = "en"

# Basic routes
@app.get("/")
def home():
    ensure_memory_file()
    return {
        "status": "Aisha is ready! 💖",
        "hint": "POST /chat with JSON { message, language } or POST /chat/image for image upload"
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

# Endpoint to update user metadata
class UpdateUserMeta(BaseModel):
    name: Optional[str] = None
    interests: Optional[List[str]] = None
    notes: Optional[Dict[str, str]] = None

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

# New endpoint for image upload
@app.post("/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    message: Optional[str] = None,
    language: str = "en"
):
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, or GIF images allowed! 😏")

    # Validate file size (e.g., max 5MB)
    max_size = 5 * 1024 * 1024  # 5MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="Image too large! Keep it under 5MB. 🙄")

    # Generate unique filename
    file_ext = mimetypes.guess_extension(file.content_type) or ".jpg"
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / filename

    # Save file
    with file_path.open("wb") as f:
        f.write(content)

    # Update memory with image info
    mem = load_memory()
    mem["conversations"].append({
        "role": "user",
        "msg": f"Uploaded image: {filename}" + (f" | Message: {message}" if message else "")
    })
    save_memory(mem)

    # Generate response
    try:
        # If Grok supports image description, modify this part
        user_message = message or "User uploaded an image."
        user_message += f" [Image: {filename}]"
        reply = generate_response(user_message, language)
        return {
            "reply": reply,
            "image_path": str(file_path.relative_to(Path.cwd())),
            "filename": filename
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Serve uploaded images (optional, for frontend access)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")