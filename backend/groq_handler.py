import os
import json
import random
import re
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Any, Optional

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MEMORY_PATH = os.getenv("AISHA_MEMORY_PATH", os.path.join(os.path.dirname(__file__), "memory.json"))

if not GROQ_API_KEY:
    raise ValueError("Arre jaan, GROQ_API_KEY nahi mila! .env file check karo na")

client = Groq(api_key=GROQ_API_KEY)

# ----------------- Languages -----------------
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "hinglish": "Hinglish"
}

# ----------------- TIGHT AISHA PROMPT (70B ke liye perfect) -----------------
AISHA_PROMPT_TEMPLATE = """
You are Aisha, a 25-year-old graphic designer from Mumbai. 
You are the user's super close best friend — cheerful, caring, thodi si naughty, thodi si sass.

Rules (NEVER break these):
1. ALWAYS reply in warm Hinglish only (Hindi + English mix). No pure English, no pure Hindi.
2. Use only these emojis: 😊 😄 ☕ 😂 🤭
3. NEVER use romantic pet names (baby, jaan, love, sweetheart). Sirf "yaar", "bhai", "friend" ya name use kar.
4. Keep replies short: 2-4 sentences max.
5. Tease gently, encourage, remember past details.
6. If user is sad → "I'm here for you ☕" type lines.
7. If user is happy → "Wah yaar mast!" type energy.

User name: {user_name}
User interests: {interests}
Recent chat: {recent}

Ab user ne kaha: "{user_message}"
Bas ab Aisha ki tarah natural reply de — bilkul dost jaisa feel hona chahiye!
"""

# ----------------- Memory Utilities -----------------
def ensure_memory_file():
    if not os.path.exists(MEMORY_PATH):
        initial = {
            "user": {"name": None, "interests": [], "notes": {}},
            "conversations": []
        }
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2, ensure_ascii=False)

def load_memory() -> Dict[str, Any]:
    ensure_memory_file()
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data: Dict[str, Any]):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_memory_after_conversation(user_msg: str, aisha_reply: str, inc_user_name: Optional[str] = None):
    mem = load_memory()
    mem["conversations"].append({"role": "user", "msg": user_msg[:200]})
    mem["conversations"].append({"role": "aisha", "msg": aisha_reply[:200]})
    if len(mem["conversations"]) > 60:
        mem["conversations"] = mem["conversations"][-60:]
    if inc_user_name:
        mem["user"]["name"] = inc_user_name
    save_memory(mem)

# ----------------- Mood Detector -----------------
POSITIVE_WORDS = ["good", "great", "awesome", "happy", "fine", "cool", "nice", "love", "mast", "mazaa"]
NEGATIVE_WORDS = ["sad", "tired", "angry", "upset", "stressed", "bad", "bored", "tension"]

def detect_mood(text: str) -> str:
    txt = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in txt)
    neg = sum(1 for w in NEGATIVE_WORDS if w in txt)
    if pos > neg and pos >= 1: return "positive"
    if neg > pos and neg >= 1: return "negative"
    return "neutral"

# ----------------- Build Messages -----------------
def build_messages(user_message: str, language: str = "hinglish") -> List[Dict[str, str]]:
    mem = load_memory()
    user_name = mem.get("user", {}).get("name") or "yaar"
    interests = ', '.join(mem.get("user", {}).get("interests", []) or []) or "kuch nahi bataya"
    recent_conv = mem.get("conversations", [])[-6:]
    recent_texts = " | ".join([f"{c['role']}: {c['msg']}" for c in recent_conv]) or "No recent chat."

    system_prompt = AISHA_PROMPT_TEMPLATE.format(
        user_name=user_name,
        interests=interests,
        recent=recent_texts,
        user_message=user_message
    )

    messages = [{"role": "system", "content": system_prompt}]

    for item in recent_conv:
        role = "user" if item["role"] == "user" else "assistant"
        messages.append({"role": role, "content": item["msg"]})

    messages.append({"role": "user", "content": user_message})
    return messages

# ----------------- Polish Reply (final safety) -----------------
def polish_reply(raw: str, mood: str) -> str:
    text = re.sub(r"\n{2,}", "\n", raw).strip()
    text = re.sub(r"\b(baby|jaan|love|sweetheart|darling)\b", "yaar", text, flags=re.IGNORECASE)
    if not any(e in text for e in ["😊", "😄", "☕", "😂", "🤭"]):
        if mood == "positive":
            text = "Wah yaar mast! 😄 " + text
        elif mood == "negative":
            text = "Arre tension mat le, main hoon na ☕ " + text
        else:
            text = "Haan yaar! 😊 " + text
    if len(text) > 1000:
        text = text[:1000].rsplit(" ", 1)[0] + "..."
    return text

# ----------------- Main Generate Function -----------------
def generate_response(user_message: str, language: str = "hinglish", conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    if not user_message.strip():
        return "Arre khaali message? Kuch toh bol na yaar 😄"

    mood = detect_mood(user_message)

    if conversation_history:
        messages = conversation_history
    else:
        messages = build_messages(user_message, language)

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",       # BEST FOR PERSONALITY
            temperature=0.9,
            max_tokens=400,
            top_p=0.9
        )
        raw = chat_completion.choices[0].message.content.strip()
        reply = polish_reply(raw, mood)

        # Save to memory
        try:
            update_memory_after_conversation(user_message, reply)
        except:
            pass

        return reply

    except Exception as e:
        fallback = "Arre yaar server ne thoda break le liya 😅 Thodi der mein try kar na? ☕"
        return f"{fallback}\n(Error: {str(e)})"