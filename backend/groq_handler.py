# groq_handler.py
import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io

# NOTE: PERSONAS comes from backend.personas
from backend.personas import PERSONAS  # ← existing in your repo

# ─────────────────────────────────────────────
# ENVIRONMENT & CLIENT SETUP
# ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found! Please check your .env file.")
client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# APP-LEVEL MEMORY (EXPORTED for main.py)
# ─────────────────────────────────────────────
APP_MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")

def ensure_memory_file():
    """Ensure top-level memory file exists (used by main.py endpoints)."""
    if not os.path.exists(APP_MEMORY_FILE):
        initial = {
            "user": {"name": None, "interests": [], "notes": {}},
            "conversations": []
        }
        # ensure dir exists (should, but be safe)
        os.makedirs(os.path.dirname(APP_MEMORY_FILE), exist_ok=True)
        with open(APP_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2, ensure_ascii=False)

def load_memory() -> Dict[str, Any]:
    """Load top-level memory.json used by routes like /memory."""
    ensure_memory_file()
    with open(APP_MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data: Dict[str, Any]):
    """Save top-level memory.json used by routes like /memory/update."""
    # atomic write pattern to avoid partial writes
    tmp = APP_MEMORY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, APP_MEMORY_FILE)

# ─────────────────────────────────────────────
# MEMORY HANDLING PER PERSONA
# ─────────────────────────────────────────────
def get_memory_path(persona_key: str = "default") -> str:
    memory_dir = os.path.join(os.path.dirname(__file__), "memory")
    os.makedirs(memory_dir, exist_ok=True)
    return os.path.join(memory_dir, f"{persona_key}.json")

def ensure_persona_memory(persona_key: str):
    path = get_memory_path(persona_key)
    if not os.path.exists(path):
        initial = {"user": {"name": None, "interests": [], "notes": {}}, "conversations": []}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2, ensure_ascii=False)

def load_persona_memory(persona_key: str) -> Dict:
    ensure_persona_memory(persona_key)
    with open(get_memory_path(persona_key), "r", encoding="utf-8") as f:
        return json.load(f)

def save_persona_memory(persona_key: str, data: Dict):
    # atomic write
    path = get_memory_path(persona_key)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# ─────────────────────────────────────────────
# IMAGE HANDLING
# ─────────────────────────────────────────────
def encode_image_to_base64(image_path: str) -> Optional[str]:
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# ─────────────────────────────────────────────
# MOOD DETECTION
# ─────────────────────────────────────────────
POSITIVE_WORDS = ["good", "great", "awesome", "happy", "cool", "fine", "love", "amazing"]
NEGATIVE_WORDS = ["sad", "tired", "angry", "upset", "stressed", "bad", "bored"]

def detect_mood(text: str) -> str:
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
def build_messages(user_message: str, persona_key: str = "default", language: str = "en", image_path: Optional[str] = None):
    mem = load_persona_memory(persona_key)
    user_name = mem.get("user", {}).get("name") or "buddy"
    interests = ', '.join(mem.get("user", {}).get("interests", []) or []) or "nothing"
    recent_conv = mem.get("conversations", [])[-6:]
    recent_texts = " | ".join([f"{c['role']}:{c['msg'][:50]}" for c in recent_conv]) or "First chat."

    system_prompt = PERSONAS.get(persona_key, PERSONAS.get("default", {"system_prompt": "You are a helpful assistant."}))["system_prompt"]

    messages = [{"role": "system", "content": system_prompt}]

    for item in recent_conv:
        role = "user" if item["role"] == "user" else "assistant"
        messages.append({"role": role, "content": item["msg"]})

    if image_path and os.path.exists(image_path):
        img_b64 = encode_image_to_base64(image_path)
        if img_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message or "Describe this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
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
    text = re.sub(r"\n{2,}", "\n", raw).strip()
    if "default" in raw.lower() or mood == "negative":
        text = re.sub(r"\b(baby|sweetheart|darling|love)\b", "buddy", text, flags=re.IGNORECASE)
    if not any(e in text for e in ["😎", "😂", "🤔", "🙄", "😏", "☕", "♡", "❤️"]):
        text += " 😎" if mood != "negative" else " ☕"
    return text[:1000]

# ─────────────────────────────────────────────
# MAIN RESPONSE GENERATOR
# ─────────────────────────────────────────────
def generate_response(user_message: str, persona_key: str = "default", language: str = "en", image_path: Optional[str] = None) -> str:
    try:
        if not user_message or not user_message.strip():
            return "Blank message? Classic move 🙄"

        mood = detect_mood(user_message)
        messages, mem_path = build_messages(user_message, persona_key, language, image_path)

        try:
            # Primary: Llama 3.3 70B
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.92,
                max_tokens=450,
                top_p=0.9
            )
            raw = chat_completion.choices[0].message.content.strip()

        except Exception as e1:
            print(f"70B failed: {e1}")
            try:
                # Fallback 1: Llama 4 Scout 17B
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=0.9,
                    max_tokens=400
                )
                raw = "[Scout mode activated] " + chat_completion.choices[0].message.content.strip()

            except Exception as e2:
                print(f"Scout also failed: {e2}")
                raw = "Arre bhai server thodi si thakan feel kar raha hai... Try again a sec later."

        # Polish + save memory
        reply = polish_reply(raw, mood)

        # Update persona memory (short preview only)
        try:
            mem = load_persona_memory(persona_key)
            mem["conversations"].append({"role": "user", "msg": user_message[:200]})
            mem["conversations"].append({"role": "assistant", "msg": reply[:200]})
            if len(mem["conversations"]) > 60:
                mem["conversations"] = mem["conversations"][-60:]
            save_persona_memory(persona_key, mem)
        except Exception as memerr:
            print(f"Memory save failed: {memerr}")

        return reply

    except Exception as e:
        print(f"Global error: {e}")
        return "Server thak gaya re baba... try again."
