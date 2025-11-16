import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found! Please check your .env file.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Mock PERSONAS (replace with actual personas module)
PERSONAS = {
    "default": {"system_prompt": "You are a friendly assistant who responds with a casual tone."}
}

# ─────────────────────────────────────────────
# MEMORY HANDLING PER PERSONA
# ─────────────────────────────────────────────
def get_memory_path(persona_key: str) -> str:
    """Generate the file path for a persona's memory file."""
    memory_dir = os.path.join(os.path.dirname(__file__), "memory")
    os.makedirs(memory_dir, exist_ok=True)
    return os.path.join(memory_dir, f"{persona_key}.json")

def ensure_persona_memory(persona_key: str) -> None:
    """Ensure a memory file exists for the given persona."""
    path = get_memory_path(persona_key)
    if not os.path.exists(path):
        initial = {"user": {"name": None, "interests": [], "notes": {}}, "conversations": []}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(initial, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to create memory file for {persona_key}: {e}")
            raise

def load_persona_memory(persona_key: str) -> Dict[str, Any]:
    """Load the memory data for a given persona."""
    try:
        ensure_persona_memory(persona_key)
        with open(get_memory_path(persona_key), "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load memory for {persona_key}: {e}")
        return {"user": {"name": None, "interests": [], "notes": {}}, "conversations": []}

def save_persona_memory(persona_key: str, data: Dict[str, Any]) -> None:
    """Save the memory data for a given persona."""
    try:
        with open(get_memory_path(persona_key), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Failed to save memory for {persona_key}: {e}")
        raise

# ─────────────────────────────────────────────
# IMAGE HANDLING
# ─────────────────────────────────────────────
def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode an image to base64 string."""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)  # Optimize image quality
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

# ─────────────────────────────────────────────
# MOOD DETECTION
# ─────────────────────────────────────────────
POSITIVE_WORDS = ["good", "great", "awesome", "happy", "cool", "fine", "love", "amazing"]
NEGATIVE_WORDS = ["sad", "tired", "angry", "upset", "stressed", "bad", "bored"]

def detect_mood(text: str) -> str:
    """Detect the mood of the input text based on keyword analysis."""
    if not text:
        return "neutral"
    txt = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in txt)
    neg = sum(1 for w in NEGATIVE_WORDS if w in txt)
    if pos > neg and pos >= 1:
        return "positive"
    if neg > pos and neg >= 1:
        return "negative"
    return "neutral"

# ─────────────────────────────────────────────
# MESSAGE BUILDING (per persona)
# ─────────────────────────────────────────────
def build_messages(
    user_message: str,
    persona_key: str = "default",
    language: str = "en",
    image_path: Optional[str] = None
) -> tuple[List[Dict[str, Any]], str]:
    """Build the message list for the Groq API based on persona and context."""
    if not persona_key in PERSONAS:
        logger.warning(f"Invalid persona_key: {persona_key}. Using default.")
        persona_key = "default"

    mem = load_persona_memory(persona_key)
    user_name = mem.get("user", {}).get("name", "buddy")
    interests = ", ".join(mem.get("user", {}).get("interests", []) or ["nothing"])
    recent_conv = mem.get("conversations", [])[-10:]  # Limit to 10 for context
    recent_texts = " | ".join([f"{c['role']}:{c['msg'][:50]}" for c in recent_conv]) or "First chat."
    logger.debug(f"Memory for {persona_key}: {recent_texts}")

    system_prompt = PERSONAS[persona_key]["system_prompt"]
    messages = [{"role": "system", "content": system_prompt}]

    # Add recent conversation history
    for item in recent_conv:
        role = "user" if item["role"] == "user" else "assistant"
        messages.append({"role": role, "content": item["msg"]})

    # Handle image if provided
    if image_path and os.path.exists(image_path):
        img_b64 = encode_image_to_base64(image_path)
        if img_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message or "Describe this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ]
            })
        else:
            messages.append({"role": "user", "content": user_message})
    else:
        messages.append({"role": "user", "content": user_message})

    return messages, get_memory_path(persona_key)

# ─────────────────────────────────────────────
# REPLY POLISHER
# ─────────────────────────────────────────────
def polish_reply(raw: str, mood: str) -> str:
    """Polish the raw response by cleaning text and adding emojis based on mood."""
    text = re.sub(r"\n{2,}", "\n", raw).strip()
    if "default" in raw.lower() or mood == "negative":
        text = re.sub(r"\b(baby|sweetheart|darling|love)\b", "buddy", text, flags=re.IGNORECASE)
    
    # Add emoji if none of the common ones are present
    if not any(e in text for e in ["😎", "😂", "🤔", "🙄", "😏", "☕"]):
        text += " 😎" if mood != "negative" else " ☕"
    
    return text[:1000]  # Truncate to prevent overly long responses

# ─────────────────────────────────────────────
# MAIN RESPONSE GENERATOR
# ─────────────────────────────────────────────
def generate_response(
    user_message: str,
    persona_key: str = "default",
    language: str = "en",
    image_path: Optional[str] = None
) -> str:
    """Generate a response for the user message using the Groq API."""
    try:
        if not user_message.strip():
            return "Blank message? Classic move 🙄"

        mood = detect_mood(user_message)
        messages, mem_path = build_messages(user_message, persona_key, language, image_path)
        logger.info(f"Generating response for persona: {persona_key}, mood: {mood}")

        # Try primary model: Llama 3.3 70B
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=1.1,
                max_tokens=450,
                top_p=0.95
            )
            raw = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Primary model failed: {e}. Falling back to Scout.")
            # Fallback: Llama 4 Scout 17B
            try:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=1.0,
                    max_tokens=400
                )
                raw = "[Scout mode activated] " + chat_completion.choices[0].message.content.strip()
            except Exception as e2:
                logger.error(f"Scout model failed: {e2}")
                return "Server thak gaya re baba... 10 sec baad try kar 😴"

        # Polish response and save to memory
        reply = polish_reply(raw, mood)
        mem = load_persona_memory(persona_key)
        mem["conversations"].append({"role": "user", "msg": user_message[:200]})
        mem["conversations"].append({"role": "assistant", "msg": reply[:200]})
        
        # Keep only the last 60 messages to prevent memory bloat
        if len(mem["conversations"]) > 60:
            mem["conversations"] = mem["conversations"][-60:]
        
        save_persona_memory(persona_key, mem)
        return reply

    except Exception as e:
        logger.error(f"Unexpected error in generate_response: {e}")
        return "Server thak gaya re baba... 10 sec baad try kar 😴"

if __name__ == "__main__":
    # Example usage
    response = generate_response("Hey, I'm feeling great today!", persona_key="default")
    print(response)