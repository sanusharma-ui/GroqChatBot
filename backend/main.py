# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq_handler import generate_response
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:5500", "http://127.0.0.1:8000", "http://127.0.0.1:5500", "*"],  # "*" temporary for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Groq Chatbot API is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    reply = generate_response(request.message)
    return {"reply": reply}

# Static files
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")