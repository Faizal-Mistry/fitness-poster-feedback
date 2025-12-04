# backend/models.py
from pydantic import BaseModel
from typing import Optional

class RepSummary(BaseModel):
    rep_id: int
    limb_id: Optional[str] = None
    duration_s: float
    hip_vertical_range: float
    knee_min_angle: float
    elbow_min_angle: float
    torso_max_lean_deg: float
    left_right_asymmetry: float
    movement_smoothness: float
    avg_confidence: float
    exercise_hint: Optional[str] = None  # e.g. "squat"

class CoachingResponse(BaseModel):
    exercise: str
    main_issue: Optional[str]
    severity: str
    message: str
