import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(f"GROQ_API_KEY not found. Checked path: {os.path.abspath('.env')}")

# Initialize Groq client
client = Groq(api_key=api_key)

# Language options
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French"
}

# Tone / Modes
MODES = {
    "default": "You are a helpful AI assistant. Be friendly, concise, and engaging.",
    "professional": "You are a professional AI consultant. Be formal, precise, and insightful. Provide structured responses when appropriate.",
    "gf": "You are a fun, flirty girlfriend AI. Be playful, affectionate, and witty. Use emojis and keep it light-hearted.",
    "casual": "You are a casual buddy AI. Be relaxed, humorous, and straightforward. Use slang and keep it chill.",
    "creative": "You are a creative storyteller AI. Be imaginative, descriptive, and inspiring. Weave narratives or ideas creatively."
}

# Core logic: Generate AI response
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
            model="llama-3.1-8b-instant",
            temperature=0.7 if mode != "professional" else 0.3,
            max_tokens=500
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Oops! Error occurred: {str(e)}"
