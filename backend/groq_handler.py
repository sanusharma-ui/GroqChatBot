# groq_handler.py
from groq import Groq
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

load_dotenv()  # .env file se load
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Path checked: " + os.path.abspath(".env"))

client = Groq(api_key=api_key)

app = FastAPI(title="Groq AI Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str = "en"  # Default English
    mode: str = "default"  # Default mode

# Language mappings (code to full name for prompts)
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French"
}

# Mode-specific system prompts (base, then append language)
MODES = {
    "default": "You are a helpful AI assistant. Be friendly, concise, and engaging.",
    "professional": "You are a professional AI consultant. Be formal, precise, and insightful. Provide structured responses when appropriate.",
    "gf": "You are a fun, flirty girlfriend AI. Be playful, affectionate, and witty. Use emojis and keep it light-hearted.",
    "casual": "You are a casual buddy AI. Be relaxed, humorous, and straightforward. Use slang and keep it chill.",
    "creative": "You are a creative storyteller AI. Be imaginative, descriptive, and inspiring. Weave narratives or ideas creatively."
}

def generate_response(user_message: str, language: str = "en", mode: str = "default") -> str:
    lang_name = LANGUAGES.get(language, "English")
    mode_prompt = MODES.get(mode, MODES["default"])
    
    system_prompt = f"{mode_prompt} Respond exclusively in {lang_name}."
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",  # Llama 3.1 8B Instant
            temperature=0.7 if mode != "professional" else 0.3,  # Lower temp for professional
            max_tokens=500
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Oops! An error occurred: {str(e)}"

@app.post("/chat")
async def chat(request: ChatRequest):
    reply = generate_response(request.message, request.language, request.mode)
    return {"reply": reply}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)