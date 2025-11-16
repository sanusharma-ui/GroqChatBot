import sys
import os
import traceback
import logging
import mimetypes
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from PIL import Image
import io
import uvicorn

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.groq_handler import (
    generate_response,
    PERSONAS,
    ensure_persona_memory,
    load_persona_memory,
    save_persona_memory
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5500,http://localhost:8000").split(",")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/uploads"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 5))
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", 60))

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Aisha — Friendly AI",
    description="Aisha: your warm, caring AI best friend. Uses Groq under the hood with persona-based modes.",
    version="2.1"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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

# Helper functions
def validate_image(content: bytes) -> bool:
    """Validate if the content is a valid image using filetype (replacement for deprecated imghdr)."""
    kind = filetype.guess(content)
    if kind:
        return kind.mime in ["image/jpeg", "image/png", "image/gif", "image/webp"]
    return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    return "".join(c for c in filename if c.isalnum() or c in [".", "_", "-"])

def cleanup_old_uploads():
    """Remove uploaded files older than 1 hour."""
    try:
        threshold = datetime.now() - timedelta(hours=1)
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < threshold.timestamp():
                file_path.unlink()
                logger.info(f"Deleted old upload: {file_path}")
    except Exception as e:
        logger.error(f"Failed to clean up uploads: {e}")

async def resize_image(content: bytes) -> bytes:
    """Resize image to reduce processing load while maintaining quality."""
    try:
        img = Image.open(io.BytesIO(content))
        img.thumbnail((1024, 1024))  # Max 1024x1024 pixels
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Failed to resize image: {e}")
        return content

# Startup event for cleanup
@app.on_event("startup")
async def startup_event():
    """Run cleanup of old uploads on startup."""
    cleanup_old_uploads()

# Routes
@app.get("/")
async def home():
    """API root endpoint."""
    try:
        ensure_persona_memory("default")
        return {
            "status": "Aisha is ready!",
            "hint": "POST /chat or /chat/image",
            "available_modes": list(PERSONAS.keys())
        }
    except Exception as e:
        logger.error(f"Error in home endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/version")
async def version():
    """Return API version."""
    return {"version": app.version}

@app.get("/memory")
async def memory():
    """Retrieve default persona memory."""
    try:
        return {"memory": load_persona_memory("default")}
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory")

@app.post("/memory/update")
async def memory_update(payload: UpdateUserMeta):
    """Update user metadata in default persona memory."""
    try:
        mem = load_persona_memory("default")
        if payload.name:
            mem["user"]["name"] = payload.name[:50]  # Limit name length
        if payload.interests:
            mem["user"]["interests"] = payload.interests[:10]  # Limit interests
        if payload.notes:
            mem["user"]["notes"].update({k: v[:200] for k, v in payload.notes.items()})  # Limit note length
        save_persona_memory("default", mem)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error updating memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to update memory")

@app.get("/modes/list")
async def list_modes():
    """List available persona modes."""
    return {"modes": {k: v["name"] for k, v in PERSONAS.items()}}

@app.post("/chat")
async def chat(request: ChatRequest, mode: str = "default", reset: bool = False):
    """Handle text-based chat requests."""
    if mode not in PERSONAS:
        mode = "default"
        logger.warning(f"Invalid mode requested, defaulting to {mode}")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message!")
    
    try:
        if reset:
            logger.info(f"Resetting memory for persona: {mode}")
            mem = {"user": {"name": None, "interests": [], "notes": {}}, "conversations": []}
            save_persona_memory(mode, mem)

        reply = generate_response(
            user_message=request.message,
            persona_key=mode,
            language=request.language
        )

        # Update memory
        mem = load_persona_memory(mode)
        mem["conversations"].append({
            "role": "user",
            "msg": request.message[:200]
        })
        mem["conversations"].append({
            "role": "assistant",
            "msg": reply[:200]
        })
        if len(mem["conversations"]) > MAX_CONVERSATION_HISTORY:
            mem["conversations"] = mem["conversations"][-MAX_CONVERSATION_HISTORY:]
        save_persona_memory(mode, mem)

        return {
            "reply": reply,
            "mode": mode,
            "display_name": PERSONAS[mode]["name"]
        }
    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/chat/image")
async def chat_image(
    file: UploadFile = File(...),
    message: Optional[str] = None,
    language: str = "en",
    mode: str = "default"
):
    """Handle image-based chat requests."""
    if mode not in PERSONAS:
        mode = "default"
        logger.warning(f"Invalid mode requested, defaulting to {mode}")

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, GIF, WebP allowed!")

    # Read and validate file
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Image too big! Max {MAX_FILE_SIZE_MB}MB.")
    if not validate_image(content):
        raise HTTPException(status_code=400, detail="Invalid image file!")

    # Resize image to reduce processing load
    content = await resize_image(content)

    # Save file
    ext = mimetypes.guess_extension(file.content_type) or ".jpg"
    filename = f"{uuid.uuid4()}{sanitize_filename(ext)}"
    file_path = UPLOAD_DIR / filename
    try:
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        raise HTTPException(status_code=500, detail="Failed to save image")

    # Generate response
    user_text = message.strip() if message and message.strip() else "Describe this image."
    try:
        reply = generate_response(
            user_message=user_text,
            persona_key=mode,
            language=language,
            image_path=str(file_path)
        )

        # Update memory
        mem = load_persona_memory(mode)
        mem["conversations"].append({
            "role": "user",
            "msg": f"[Image: {filename}] {user_text}"[:200]
        })
        mem["conversations"].append({
            "role": "assistant",
            "msg": reply[:200]
        })
        if len(mem["conversations"]) > MAX_CONVERSATION_HISTORY:
            mem["conversations"] = mem["conversations"][-MAX_CONVERSATION_HISTORY:]
        save_persona_memory(mode, mem)

        return {
            "reply": reply,
            "image_path": f"uploads/{filename}",
            "filename": filename,
            "mode": mode,
            "display_name": PERSONAS[mode]["name"]
        }
    except Exception as e:
        logger.error(f"Image chat error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Vision error: {str(e)}")
    finally:
        # Delete file immediately after processing
        try:
            file_path.unlink()
            logger.info(f"Deleted temporary image: {file_path}")
        except Exception:
            pass

# Serve uploaded images statically
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)