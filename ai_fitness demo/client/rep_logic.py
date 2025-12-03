# client/rep_logic.py
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class RepState:
    state: str = "EXTENDED"        # "EXTENDED" or "FLEXED"
    rep_id: int = 0
    rep_start_time: Optional[float] = None

    hip_positions: list = field(default_factory=list)
    knee_angles: list = field(default_factory=list)
    elbow_angles: list = field(default_factory=list)
    torso_devs: list = field(default_factory=list)
    confidences: list = field(default_factory=list)

    last_rep_end_time: Optional[float] = None


# Per-exercise configuration
EXERCISE_CONFIG = {
    "squat": {
        "primary_joint": "knee",
        "flexed_threshold": 35.0,   # need decent bend
        "extended_threshold": 15.0  # back to almost straight
    },
    "pushup": {
        "primary_joint": "elbow",
        "flexed_threshold": 40.0,
        "extended_threshold": 20.0
    },
    "bicep_curl": {
        "primary_joint": "elbow",
        "flexed_threshold": 90.0,
        "extended_threshold": 20.0
    },
    # NEW: Lunge
    "lunge": {
        "primary_joint": "knee",
        # For lunge, we want a clear bend in the front knee
        # knee_flex = 180 - knee_angle, so > 40 means angle < 140 deg
        "flexed_threshold": 40.0,
        # Extended when knee almost straight again
        "extended_threshold": 15.0
    },
    # NEW: Mountain climber
    "mountain_climber": {
        "primary_joint": "knee",
        # Faster, smaller movement is okay, but still want visible flex
        "flexed_threshold": 45.0,
        # Treat leg as "back out" when fairly straight again
        "extended_threshold": 20.0
    }
}


DEFAULT_CONFIG = {
    "primary_joint": "knee",
    "flexed_threshold": 35.0,
    "extended_threshold": 15.0
}


def get_exercise_config(exercise_name: Optional[str]) -> Dict[str, Any]:
    if exercise_name and exercise_name in EXERCISE_CONFIG:
        return EXERCISE_CONFIG[exercise_name]
    return DEFAULT_CONFIG


def update_rep_state(
    rep_state: RepState,
    features: Dict[str, Any],
    avg_confidence: float,
    exercise_hint: Optional[str] = None
):
    """
    Generic rep detection:
      - uses flex of primary joint (knee or elbow)
      - counts rep when we go: EXTENDED -> FLEXED -> EXTENDED
    """
    now = time.time()

    # These keys are the same as your existing pipeline
    knee_angle = features["knee_min_angle_frame"]
    elbow_angle = features["elbow_min_angle_frame"]
    hip_y      = features["center_hip_y"]
    torso_dev  = features["torso_dev_frame"]

    # joint bending amounts (0 = straight, higher = more bent)
    knee_flex  = 180.0 - knee_angle
    elbow_flex = 180.0 - elbow_angle

    config = get_exercise_config(exercise_hint)
    primary_joint = config["primary_joint"]
    flexed_threshold = config["flexed_threshold"]
    extended_threshold = config["extended_threshold"]

    if primary_joint == "knee":
        flex_amount = knee_flex
    else:
        flex_amount = elbow_flex

    rep_completed = False
    rep_summary = None

    if rep_state.state == "EXTENDED":
        # Wait for bend enough to start rep
        if flex_amount > flexed_threshold:
            rep_state.state = "FLEXED"
            rep_state.rep_start_time = now
            rep_state.hip_positions = [hip_y]
            rep_state.knee_angles = [knee_angle]
            rep_state.elbow_angles = [elbow_angle]
            rep_state.torso_devs = [torso_dev]
            rep_state.confidences = [avg_confidence]

    elif rep_state.state == "FLEXED":
        # Accumulate while in rep
        rep_state.hip_positions.append(hip_y)
        rep_state.knee_angles.append(knee_angle)
        rep_state.elbow_angles.append(elbow_angle)
        rep_state.torso_devs.append(torso_dev)
        rep_state.confidences.append(avg_confidence)

        # Rep ends when joint is nearly straight again
        if flex_amount < extended_threshold:
            rep_state.state = "EXTENDED"
            rep_state.rep_id += 1
            end_time = now
            duration = end_time - (rep_state.rep_start_time or end_time)

            hip_vertical_range = 0.0
            if rep_state.hip_positions:
                hip_vertical_range = max(rep_state.hip_positions) - min(rep_state.hip_positions)

            rep_summary = {
                "rep_id": rep_state.rep_id,
                "duration_s": float(duration),
                "hip_vertical_range": float(hip_vertical_range),
                "knee_min_angle": float(min(rep_state.knee_angles)) if rep_state.knee_angles else 180.0,
                "elbow_min_angle": float(min(rep_state.elbow_angles)) if rep_state.elbow_angles else 180.0,
                "torso_max_lean_deg": float(max(rep_state.torso_devs)) if rep_state.torso_devs else 0.0,
                "left_right_asymmetry": 0.0,  # later: difference between left/right
                "movement_smoothness": 0.8,   # later: compute from velocity variance
                "avg_confidence": float(np.mean(rep_state.confidences)) if rep_state.confidences else 0.0,
                "exercise_hint": exercise_hint,
            }

            rep_state.last_rep_end_time = end_time
            rep_completed = True

    return rep_completed, rep_summary
