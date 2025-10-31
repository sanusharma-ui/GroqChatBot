from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from backend.groq_handler import generate_response
import os

app = FastAPI(title="Groq AI Chatbot")

# Allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
        "https://sonu-frontend.onrender.com/",
        "*"  # for Render or testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for POST
class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    mode: str = "default"

@app.get("/")
def home():
    return {"message": "Groq Chatbot API is live and running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    reply = generate_response(request.message, request.language, request.mode)
    return {"reply": reply}

# Mount frontend (for static hosting)
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
