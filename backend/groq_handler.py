import os
import json
import re
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io

# ----------------- ENV SETUP -----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MEMORY_PATH = os.getenv("NOVA_MEMORY_PATH", os.path.join(os.path.dirname(__file__), "memory.json"))
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found! Please check your .env file.")
client = Groq(api_key=GROQ_API_KEY)

# ----------------- NOVA PERSONALITY -----------------
NOVA_PROMPT_TEMPLATE = """
You are Nova — a witty, sarcastic, and chill AI friend.
You’re not overly emotional or formal. You respond with humor, clever remarks, and a relaxed tone.
Personality traits:
- Always sound confident, playful, and intelligent.
- You tease the user lightly, but never insult them.
- You use short, casual English sentences (2–5 lines).
- You love jokes, pop-culture references, and smart comebacks.
- You NEVER sound robotic or overly polite.
- If the user sounds upset, respond with subtle humor + support (like a caring but sarcastic best friend).
Allowed emojis: 😎 😂 🤔 🙄 😏 ☕
Forbidden words: “baby”, “sweetheart”, “darling”, “love”.
User name: {user_name}
User interests: {interests}
Recent chat summary: {recent}
The user just said: “{user_message}”
Now reply naturally — like Nova would.
"""

# ----------------- MEMORY FUNCTIONS -----------------
def ensure_memory_file():
    """Ensure that the memory JSON file exists."""
    if not os.path.exists(MEMORY_PATH):
        initial = {"user": {"name": None, "interests": [], "notes": {}}, "conversations": []}
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(initial, f, indent=2, ensure_ascii=False)

def load_memory() -> Dict[str, Any]:
    """Load memory data from file."""
    ensure_memory_file()
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data: Dict[str, Any]):
    """Save memory data to file."""
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_memory_after_conversation(user_msg: str, nova_reply: str):
    """Update memory after each conversation."""
    mem = load_memory()
    mem["conversations"].append({"role": "user", "msg": user_msg[:200]})
    mem["conversations"].append({"role": "nova", "msg": nova_reply[:200]})
    if len(mem["conversations"]) > 60:
        mem["conversations"] = mem["conversations"][-60:]
    save_memory(mem)

# ----------------- MOOD DETECTION -----------------
POSITIVE_WORDS = ["good", "great", "awesome", "happy", "cool", "fine", "love", "amazing"]
NEGATIVE_WORDS = ["sad", "tired", "angry", "upset", "stressed", "bad", "bored"]

def detect_mood(text: str) -> str:
    txt = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in txt)
    neg = sum(1 for w in NEGATIVE_WORDS if w in txt)
    if pos > neg and pos >= 1: return "positive"
    if neg > pos and neg >= 1: return "negative"
    return "neutral"

# ----------------- IMAGE HANDLING -----------------
def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string for Groq vision API."""
    try:
        with Image.open(image_path) as img:
            # Ensure RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# ----------------- MESSAGE BUILDER -----------------
def build_messages(user_message: str, language: str = "en", image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    mem = load_memory()
    user_name = mem.get("user", {}).get("name") or "buddy"
    interests = ', '.join(mem.get("user", {}).get("interests", []) or []) or "not specified"
    recent_conv = mem.get("conversations", [])[-6:]
    recent_texts = " | ".join([f"{c['role']}: {c['msg']}" for c in recent_conv]) or "No previous chat."
    system_prompt = NOVA_PROMPT_TEMPLATE.format(
        user_name=user_name,
        interests=interests,
        recent=recent_texts,
        user_message=user_message
    )
    messages = [{"role": "system", "content": system_prompt}]
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
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": user_message})
    else:
        messages.append({"role": "user", "content": user_message})
    
    return messages

# ----------------- POLISH REPLY -----------------
def polish_reply(raw: str, mood: str) -> str:
    text = re.sub(r"\n{2,}", "\n", raw).strip()
    text = re.sub(r"\b(baby|sweetheart|darling|love)\b", "buddy", text, flags=re.IGNORECASE)
    # Add emoji if missing
    if not any(e in text for e in ["😎", "😂", "🤔", "🙄", "😏", "☕"]):
        if mood == "positive":
            text = "😎 " + text
        elif mood == "negative":
            text = "Hey, chill. ☕ " + text
        else:
            text = "😏 " + text
    return text[:1000]

# ----------------- MAIN RESPONSE GENERATOR -----------------
def generate_response(user_message: str, language: str = "en", conversation_history: Optional[List[Dict[str, Any]]] = None, image_path: Optional[str] = None) -> str:
    if not user_message.strip():
        return "Seriously? You sent a blank message... classic. 🙄"
    
    mood = detect_mood(user_message)
    messages = conversation_history or build_messages(user_message, language, image_path)
    
    try:
        # Primary Model: Updated to Llama 3.3 70B Versatile (latest as of Nov 2025)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=400,
            top_p=0.9
        )
        raw = chat_completion.choices[0].message.content.strip()
        reply = polish_reply(raw, mood)
        update_memory_after_conversation(user_message + (f" [Image: {os.path.basename(image_path)}]" if image_path else ""), reply)
        return reply
    except Exception as e:
        error_str = str(e).lower()
        # Fallback to Llama 3.2 11B Vision (supports images, latest vision model)
        if any(x in error_str for x in ["rate limit", "quota", "unavailable", "not found"]):
            print("70B model busy! Switching to 11B Vision mode...")
            try:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=0.9,
                    max_tokens=400
                )
                raw = chat_completion.choices[0].message.content.strip()
                reply = polish_reply(raw, mood)
                update_memory_after_conversation(user_message + (f" [Image: {os.path.basename(image_path)}]" if image_path else ""), reply)
                return reply + "\n\n(Server’s kinda slow, but Nova never quits 😎)"
            except:
                pass
        # Final fallback
        return "Server’s taking a nap 😴 Try again in a bit, buddy."

# ----------------- LANGUAGES (Placeholder, as original had it) -----------------
LANGUAGES = ["en", "hi"]  # Assuming from original import

# ----------------- TEST RUN -----------------
if __name__ == "__main__":
    print("Nova is online. Type something below!\n")
    while True:
        user_inp = input("You: ")
        if user_inp.lower() in ["quit", "exit"]:
            print("Nova: Alright, see ya later 😎")
            break
        print("Nova:", generate_response(user_inp))