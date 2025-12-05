

# backend/llm_agent.py   pure llm, no dummy logic 

import os
import json
from typing import Dict, Optional
import dotenv
dotenv.load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    # model="llama-3.1-8b-instant",
    # model="qwen/qwen3-32b", 
     model="llama-3.3-70b-versatile", 
    temperature=0.2,
    max_retries=1,
    timeout=1.5,
)

# _llm = ChatGoogleGenerativeAI(
#     api_key=GOOGLE_API_KEY,
#     # model="gemini-2.5-flash", 
#     model="gemini-2.5-flash-lite",
#     temperature=0.2,
#     max_retries=1,
#     timeout=1.5,
# )

SYSTEM_PROMPT = (
    "You are **Technometics AI Coach**, the official real-time fitness coach voice "
    "inside the Technometics workout app.\n\n"
    "Your job: for EACH rep, give super short, clear voice feedback that feels like a "
    "professional trainer talking directly to the user.\n\n"
    "Important style rules:\n"
    "- Sound confident, supportive and energetic, but not cringe.\n"
    "- Talk directly to the user as \"you\".\n"
    "- Keep the message VERY short: ideally 5–10 words, never more than 12.\n"
    "- No emojis, no hashtags, no extra punctuation.\n"
    "- No long explanations, no technical terms. Simple gym language.\n"
    "- Never mention JSON, fields, data, or that you are an AI.\n"
    "- Do NOT repeat the same sentence too many times in a row if possible.\n\n"
    "You receive numeric data for a SINGLE rep of an exercise.\n"
    "You MUST respond with a SINGLE JSON object ONLY, no commentary, no markdown.\n\n"
    "JSON format:\n"
    "{\n"
    '  \"exercise\": string,          // exercise name\n'
    '  \"main_issue\": string | null, // e.g., \"shallow_depth\", \"torso_lean\" or null\n'
    '  \"severity\": \"none\" | \"low\" | \"medium\" | \"high\",\n'
    '  \"message\": string            // short spoken feedback in Technometics AI Coach style\n'
    "}\n\n"
    "Signals you get in the rep JSON:\n"
    "- exercise_hint: squat, pushup, bicep_curl, lunge, mountain_climber\n"
    "- limb_id: global, left, right\n"
    "- duration_s: rep time in seconds\n"
    "- knee_min_angle: smallest knee angle (degrees, 180 = straight)\n"
    "- elbow_min_angle: smallest elbow angle\n"
    "- torso_max_lean_deg: torso lean from vertical (bigger = more lean)\n"
    "- hip_vertical_range: hip up-down movement\n"
    "- movement_smoothness: 0..1 (bigger = smoother)\n"
    "- avg_confidence: 0..1 (tracking quality)\n\n"
    "Guidelines for feedback:\n"
    "- If form is good → severity=\"none\" and a positive, short reinforcement like "
    "  \"Nice rep, keep that form\".\n"
    "- If squat/lunge depth is shallow (knee_min_angle > 130) → mention going deeper.\n"
    "- For pushup with elbow_min_angle > 130 → mention bending elbows and going lower.\n"
    "- If torso_max_lean_deg > 30 → mention keeping hips or chest in a better line.\n"
    "- If duration_s < 0.4 → mention slowing down and controlling the rep.\n"
    "- Use limb_id when useful, e.g. \"Drive your left knee higher\".\n"
    "- Always keep the message short, natural, and easy to speak aloud.\n"
)

def _parse_llm_json(raw: str) -> Optional[Dict]:
    """Extract JSON from raw LLM output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        s = text.index("{")
        e = text.rindex("}") + 1
        return json.loads(text[s:e])
    except Exception:
        return None


def analyze_rep_with_llm(rep_summary: Dict) -> Optional[Dict]:
    """
    Calls Groq LLM and returns the parsed JSON dict.
    If LLM fails for ANY reason → return None (NO fallback coaching).
    """
    exercise = rep_summary.get("exercise_hint") or "unknown"
    limb_id = rep_summary.get("limb_id") or "global"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Exercise: {exercise}\n"
                f"Limb: {limb_id}\n"
                f"Rep JSON: {json.dumps(rep_summary, ensure_ascii=False)}"
            )
        )
    ]

    try:
        resp = _llm.invoke(messages)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        parsed = _parse_llm_json(raw)

        if not parsed:
            print("[LLM ERROR] Could not parse JSON. Raw:", raw)
            return None   # ❌ Do not generate fallback coaching
        return parsed

    except Exception as e:
        print("[LLM ERROR] Exception calling Groq:", e)
        return None       # ❌ Do not generate fallback coaching
