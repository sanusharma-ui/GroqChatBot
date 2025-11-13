import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io
from backend.personas import PERSONAS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ENVIRONMENT & CLIENT SETUP
# ─────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found! Please check your .env file.")
client = Groq(api_key=GROQ_API_KEY)

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
    with open(get_memory_path(persona_key), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
        logger.error(f"Error encoding image: {e}")
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
    recent_conv = mem.get("conversations", [])[-10:]  # Increased to 10 messages for better context
    recent_texts = " | ".join([f"{c['role']}:{c['msg'][:50]}" for c in recent_conv]) or "First chat."
    logger.info(f"Memory for {persona_key}: {recent_texts}")
    system_prompt = PERSONAS.get(persona_key, PERSONAS["default"])["system_prompt"]
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
    if not any(e in text for e in ["😎", "😂", "🤔", "🙄", "😏", "☕"]):
        text += " 😎" if mood != "negative" else " ☕"
    return text[:1000]

# ─────────────────────────────────────────────
# MAIN RESPONSE GENERATOR
# ─────────────────────────────────────────────
def generate_response(user_message: str, persona_key: str = "default", language: str = "en", image_path: Optional[str] = None) -> str:
    try:
        if not user_message.strip():
            return "Blank message? Classic move 🙄"
        mood = detect_mood(user_message)
        messages, mem_path = build_messages(user_message, persona_key, language, image_path)
        logger.info(f"Using persona: {persona_key}, System prompt: {PERSONAS[persona_key]['system_prompt'][:50]}...")
        try:
            # Primary: Llama 3.3 70B
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=1.1,
                max_tokens=450,
                top_p=0.95
            )
            raw = chat_completion.choices[0].message.content.strip()
        except Exception as e1:
            logger.error(f"70B failed: {e1}")
            try:
                # Fallback 1: Llama 4 Scout 17B
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=1.0,
                    max_tokens=400
                )
                raw = "[Scout mode activated] " + chat_completion.choices[0].message.content.strip()
            except Exception as e2:
                logger.error(f"Scout also failed: {e2}")
                raw = "Arre bhai server thodi si thakan feel kar raha hai... 10 second baad try kar na? Main abhi bhi yahin hoon"
        # Polish + save memory
        reply = polish_reply(raw, mood)
        mem = load_persona_memory(persona_key)
        mem["conversations"].append({"role": "user", "msg": user_message[:200]})
        mem["conversations"].append({"role": "assistant", "msg": reply[:200]})
        if len(mem["conversations"]) > 60:
            mem["conversations"] = mem["conversations"][-60:]
        save_persona_memory(persona_key, mem)
        return reply
    except Exception as e:
        logger.error(f"Global error: {e}")
        return "Server thak gaya re baba... 10 sec baad try kar 😴"