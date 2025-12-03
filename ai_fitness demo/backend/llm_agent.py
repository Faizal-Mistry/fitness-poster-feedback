# backend/llm_agent.py
import json
from typing import Dict

"""
TEMP VERSION: No real LLM calls.
Uses only dummy_coaching() so backend is always fast and never times out.

Once the end-to-end demo is stable, we can plug in Qwen/Gemini here.
"""

def dummy_coaching(rep_summary: Dict) -> Dict:
    """
    Simple fallback coaching logic based only on numeric features.
    This is fast and runs locally.
    """
    exercise = (rep_summary.get("exercise_hint") or "unknown").lower()
    knee = rep_summary.get("knee_min_angle", 180.0)
    elbow = rep_summary.get("elbow_min_angle", 180.0)
    torso = rep_summary.get("torso_max_lean_deg", 0.0)
    duration = rep_summary.get("duration_s", 1.0)
    hip_range = rep_summary.get("hip_vertical_range", 0.0)

    # Squat
    if exercise == "squat":
        if knee > 135:
            return {
                "exercise": "squat",
                "main_issue": "shallow_depth",
                "severity": "low",
                "message": "Go a bit deeper in your squat."
            }
        if torso > 30:
            return {
                "exercise": "squat",
                "main_issue": "leaning_forward",
                "severity": "medium",
                "message": "Keep your chest up and reduce forward lean."
            }

    # Pushup
    if exercise == "pushup":
        if elbow > 130:
            return {
                "exercise": "pushup",
                "main_issue": "shallow_depth",
                "severity": "low",
                "message": "Bend elbows more and lower your chest."
            }
        if torso > 25:
            return {
                "exercise": "pushup",
                "main_issue": "hips_sagging",
                "severity": "medium",
                "message": "Keep your body in a straight line."
            }

    # Bicep curl
    if exercise == "bicep_curl":
        if elbow > 80:
            return {
                "exercise": "bicep_curl",
                "main_issue": "shallow_depth",
                "severity": "low",
                "message": "Curl higher to fully contract your biceps."
            }
        if duration < 0.3:
            return {
                "exercise": "bicep_curl",
                "main_issue": "too_fast",
                "severity": "low",
                "message": "Slow down and control each curl."
            }

    # Lunge
    if exercise == "lunge":
        if knee > 135:
            return {
                "exercise": "lunge",
                "main_issue": "shallow_depth",
                "severity": "low",
                "message": "Drop your back knee lower for a deeper lunge."
            }
        if hip_range < 4.0:
            return {
                "exercise": "lunge",
                "main_issue": "not_moving_enough",
                "severity": "low",
                "message": "Step a bit bigger and lower into the lunge."
            }
        if torso > 30:
            return {
                "exercise": "lunge",
                "main_issue": "leaning_forward",
                "severity": "medium",
                "message": "Keep your torso more upright in the lunge."
            }

    # Mountain climber
    if exercise == "mountain_climber":
        if knee > 110:
            return {
                "exercise": "mountain_climber",
                "main_issue": "short_range",
                "severity": "low",
                "message": "Drive your knee closer toward your chest."
            }
        if duration < 0.2:
            return {
                "exercise": "mountain_climber",
                "main_issue": "too_fast",
                "severity": "low",
                "message": "Control the movement, not just speed."
            }
        if torso > 25:
            return {
                "exercise": "mountain_climber",
                "main_issue": "hips_too_high",
                "severity": "medium",
                "message": "Keep your hips lower in a plank position."
            }

    # Generic checks (apply to any exercise)
    if torso > 35:
        return {
            "exercise": exercise,
            "main_issue": "leaning_forward",
            "severity": "medium",
            "message": "Reduce torso swing and stay upright."
        }

    if duration < 0.4:
        return {
            "exercise": exercise,
            "main_issue": "too_fast",
            "severity": "low",
            "message": "Slow down a bit and control the movement."
        }

    return {
        "exercise": exercise,
        "main_issue": None,
        "severity": "none",
        "message": "Great rep, keep going!"
    }


def analyze_rep_with_llm(rep_summary: Dict) -> Dict:
    """
    TEMP: just call dummy_coaching.
    Later we'll connect this to a real LLM (Qwen/Gemini).
    """
    return dummy_coaching(rep_summary)
