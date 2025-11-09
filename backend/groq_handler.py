
import os
import json
import random
import re
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Any, Optional

# ----------------- Load config -----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MEMORY_PATH = os.getenv("AISHA_MEMORY_PATH", os.path.join(os.path.dirname(__file__), "memory.json"))

if not GROQ_API_KEY:
    raise ValueError("Arre jaan, GROQ_API_KEY nahi mila! .env file check karo na 😅")

client = Groq(api_key=GROQ_API_KEY)

# ----------------- Languages & Mode -----------------
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "hinglish": "Hinglish"
}

# Aisha personality — Groq-friendly (no explicit romantic pet-names)
AISHA_PROMPT = """
You are Aisha, a 25-year-old graphic designer from Mumbai who is the user's cheerful and caring best friend.
Speak in a warm, playful, friendly tone with light Hinglish (mix Hindi and English).
You tease gently, encourage, and remember small facts from past chats to personalize replies.
Do NOT use explicit romantic pet names like 'baby' or 'love'; instead show care as a close friend.
Use emojis sparingly (😊, 😄, ☕). If asked for help, give accurate answers wrapped in friendly style.
When available, incorporate details from the conversation history and memory.
"""

# ----------------- Simple Memory Utilities -----------------
def ensure_memory_file():
    if not os.path.exists(MEMORY_PATH):
        initial = {
            "user": {
                "name": None,
                "interests": [],
                "notes": {}
            },
            "conversations": []  # list of short {role,msg,ts}
        }
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2)

def load_memory() -> Dict[str, Any]:
    ensure_memory_file()
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data: Dict[str, Any]):
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_memory_after_conversation(user_msg: str, aisha_reply: str, inc_user_name: Optional[str] = None):
    mem = load_memory()
    # store last messages (short)
    mem["conversations"].append({"role": "user", "msg": user_msg})
    mem["conversations"].append({"role": "aisha", "msg": aisha_reply})
    # keep only last 30 entries to avoid too-large context
    if len(mem["conversations"]) > 60:
        mem["conversations"] = mem["conversations"][-60:]
    # optional: capture a name if user said "my name is X"
    if inc_user_name:
        mem["user"]["name"] = inc_user_name
    save_memory(mem)

# ----------------- Micro Mood Detector (very light) -----------------
POSITIVE_WORDS = ["good", "great", "awesome", "happy", "fine", "cool", "nice", "love"]
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

# ----------------- Post-process assistant reply to gentle Aisha style -----------------
def polish_reply_for_aisha(raw: str, mood_hint: str = "neutral") -> str:
    """Ensure reply follows friendly tone; avoid forbidden pet-names."""
    # Strip excessive newlines
    text = re.sub(r"\n{2,}", "\n", raw).strip()
    # Replace any accidental romantic pet-names (conservative)
    text = re.sub(r"\b(baby|sweetheart|darling|love)\b", "friend", text, flags=re.IGNORECASE)
    # If model didn't include any emoji or friendly line, add a small friendly prefix/suffix
    if "😊" not in text and "😄" not in text and "☕" not in text:
        if mood_hint == "positive":
            text = "Aww that's great! 😊 " + text
        elif mood_hint == "negative":
            text = "Oh no, I'm here for you. ☕ " + text
        else:
            text = "Hey! " + text
    # Cap length (sensible)
    if len(text) > 1200:
        text = text[:1200].rsplit(" ", 1)[0] + "..."
    return text

# ----------------- Build messages (system + memory + convo) -----------------
def build_messages(user_message: str, language: str = "en") -> List[Dict[str, str]]:
    # Get memory
    mem = load_memory()
    user_name = mem.get("user", {}).get("name") or "friend"
    recent_conv = mem.get("conversations", [])[-6:]  # small recent history

    # Create system prompt with limited memory summary
    memory_summary = f"User name: {user_name}. Interests: {', '.join(mem.get('user', {}).get('interests', []) or [])}."
    recent_texts = " | ".join([f"{c['role']}: {c['msg']}" for c in recent_conv]) or "No recent conv."

    system_prompt = f"{AISHA_PROMPT}\nMemory summary: {memory_summary}\nRecent: {recent_texts}\nRespond in a warm, friendly way in {LANGUAGES.get(language,'English')}."

    messages = [{"role": "system", "content": system_prompt}]
    # append recent conv as user/assistant messages to preserve context
    for item in recent_conv:
        # map role names to model roles
        role = "user" if item["role"] == "user" else "assistant"
        messages.append({"role": role, "content": item["msg"]})
    # finally the new user input
    messages.append({"role": "user", "content": user_message})
    return messages

# ----------------- Main generate function -----------------
def generate_response(user_message: str, language: str = "en", conversation_history: Optional[List[Dict[str,str]]] = None) -> str:
    # small guard
    if not user_message or not user_message.strip():
        return "Arre, kya khaali message bheja? Kuch likh ke bhej na 😊"

    # mood hint for post-processing
    mood = detect_mood(user_message)

    # build messages (if conversation_history passed, prefer that for local dev)
    if conversation_history:
        messages = conversation_history
    else:
        messages = build_messages(user_message, language)

    try:
        # call Groq
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.8,
            max_tokens=500
        )
        raw = chat_completion.choices[0].message.content.strip()
        polished = polish_reply_for_aisha(raw, mood_hint=mood)

        # update memory (non-blocking friendly)
        try:
            update_memory_after_conversation(user_message, polished)
        except Exception:
            # don't fail on memory errors
            pass

        return polished

    except Exception as e:
        # Friendly fallback message
        fallback = "Arre, kuch gadbad ho gaya — server ne thoda break le liya. Chal main try karti hoon fir se? ☕"
        return f"{fallback}\n(Error: {str(e)})"
