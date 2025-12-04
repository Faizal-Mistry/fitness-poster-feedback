# client/rep_logic_multi.py

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class LimbState:
    """
    State machine for a single limb:
      EXTENDED -> FLEXED -> EXTENDED = 1 rep
    """
    state: str = "EXTENDED"        # "EXTENDED" or "FLEXED"
    rep_id: int = 0
    rep_start_time: Optional[float] = None

    hip_positions: list = field(default_factory=list)
    joint_angles: list = field(default_factory=list)   # angle used for this limb
    torso_devs: list = field(default_factory=list)
    confidences: list = field(default_factory=list)

    last_rep_end_time: Optional[float] = None


@dataclass
class MultiRepState:
    """
    Holds per-limb states for the current exercise.
    Keys are limb IDs like "global", "left", "right".
    """
    limb_states: Dict[str, LimbState] = field(default_factory=dict)


# 🧩 Per-exercise configuration
# - limbs: which logical limbs are involved
# - joint_source: which feature key to use for each limb
# - thresholds are on flex_amount = 180 - angle
EXERCISE_CONFIG: Dict[str, Dict[str, Any]] = {
    # 1) Classic symmetric exercises: use a single "global" limb
    "squat": {
        "limbs": ["global"],
        "joint_source": {
            "global": "knee_min_angle_frame",
        },
        "flexed_threshold": 35.0,
        "extended_threshold": 15.0,
        "min_interval_s": 0.35,
    },
    "pushup": {
        "limbs": ["global"],
        "joint_source": {
            "global": "elbow_min_angle_frame",
        },
        "flexed_threshold": 40.0,
        "extended_threshold": 20.0,
        "min_interval_s": 0.35,
    },
    "bicep_curl": {
        "limbs": ["global"],
        "joint_source": {
            "global": "elbow_min_angle_frame",
        },
        "flexed_threshold": 90.0,
        "extended_threshold": 20.0,
        "min_interval_s": 0.25,
    },
    "lunge": {
        "limbs": ["global"],
        "joint_source": {
            "global": "knee_min_angle_frame",
        },
        "flexed_threshold": 40.0,
        "extended_threshold": 15.0,
        "min_interval_s": 0.35,
    },

    # 2) Alternating exercise: mountain climber
    #    Each leg is treated as its own "limb".
    "mountain_climber": {
        "limbs": ["left", "right"],
        "joint_source": {
            "left": "left_knee_angle_frame",
            "right": "right_knee_angle_frame",
        },
        "flexed_threshold": 60.0,   # knee driven forward
        "extended_threshold": 25.0, # knee back/extended
        "min_interval_s": 0.10,     # allow fast cadence
    },
}


DEFAULT_CONFIG = {
    "limbs": ["global"],
    "joint_source": {"global": "knee_min_angle_frame"},
    "flexed_threshold": 35.0,
    "extended_threshold": 15.0,
    "min_interval_s": 0.3,
}


def get_exercise_config(exercise_name: Optional[str]) -> Dict[str, Any]:
    if exercise_name and exercise_name in EXERCISE_CONFIG:
        return EXERCISE_CONFIG[exercise_name]
    return DEFAULT_CONFIG


def update_multi_rep_state(
    multi_state: MultiRepState,
    features: Dict[str, Any],
    avg_confidence: float,
    exercise_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generic multi-limb rep detection:
      - Works for symmetric exercises (1 limb "global")
      - Works for alternating exercises (limbs ["left","right"], etc.)
      - Returns a list of rep_summaries (0, 1, or more) for limbs that
        completed a rep in this frame.

    rep_summary is compatible with your backend:
      {
        "rep_id": int,
        "limb_id": str,
        "duration_s": float,
        "hip_vertical_range": float,
        "knee_min_angle": float,
        "elbow_min_angle": float,
        "torso_max_lean_deg": float,
        "left_right_asymmetry": float,
        "movement_smoothness": float,
        "avg_confidence": float,
        "exercise_hint": str,
      }
    """
    now = time.time()
    cfg = get_exercise_config(exercise_hint)

    limbs: List[str] = cfg["limbs"]
    joint_source: Dict[str, str] = cfg["joint_source"]
    flexed_threshold: float = cfg["flexed_threshold"]
    extended_threshold: float = cfg["extended_threshold"]
    min_interval_s: float = cfg.get("min_interval_s", 0.0)

    # Ensure states exist for all limbs
    for limb_id in limbs:
        if limb_id not in multi_state.limb_states:
            multi_state.limb_states[limb_id] = LimbState()

    completed_reps: List[Dict[str, Any]] = []

    # Shared body features
    center_hip_y = features["center_hip_y"]
    torso_dev = features["torso_dev_frame"]

    for limb_id in limbs:
        state = multi_state.limb_states[limb_id]

        # Which angle drives this limb?
        angle_feature_name = joint_source[limb_id]
        driver_angle = float(features[angle_feature_name])
        flex_amount = 180.0 - driver_angle  # 0 = straight, higher = more bent

        # Time since last rep ended for this limb
        if state.last_rep_end_time is None:
            time_since_last = 999.0
        else:
            time_since_last = now - state.last_rep_end_time

        # ------------------ STATE MACHINE PER LIMB ------------------
        if state.state == "EXTENDED":
            # Start rep when we bend enough AND waited enough time
            if flex_amount > flexed_threshold and time_since_last >= min_interval_s:
                state.state = "FLEXED"
                state.rep_start_time = now
                state.hip_positions = [center_hip_y]
                state.joint_angles = [driver_angle]
                state.torso_devs = [torso_dev]
                state.confidences = [avg_confidence]

        elif state.state == "FLEXED":
            # Accumulate while in rep
            state.hip_positions.append(center_hip_y)
            state.joint_angles.append(driver_angle)
            state.torso_devs.append(torso_dev)
            state.confidences.append(avg_confidence)

            # End rep when we return near extended position
            if flex_amount < extended_threshold:
                state.state = "EXTENDED"
                state.rep_id += 1
                end_time = now
                duration = end_time - (state.rep_start_time or end_time)

                hip_vertical_range = 0.0
                if state.hip_positions:
                    hip_vertical_range = max(state.hip_positions) - min(state.hip_positions)

                joint_min_angle = float(min(state.joint_angles)) if state.joint_angles else 180.0
                torso_max_lean_deg = float(max(state.torso_devs)) if state.torso_devs else 0.0
                avg_conf = float(np.mean(state.confidences)) if state.confidences else 0.0

                # Map joint_min_angle to knee_min_angle / elbow_min_angle
                # so your backend stays compatible
                angle_key = angle_feature_name.lower()
                if "knee" in angle_key:
                    knee_min_angle = joint_min_angle
                    elbow_min_angle = 180.0
                elif "elbow" in angle_key:
                    knee_min_angle = 180.0
                    elbow_min_angle = joint_min_angle
                else:
                    # Fallback: treat as knee
                    knee_min_angle = joint_min_angle
                    elbow_min_angle = 180.0

                rep_summary = {
                    "rep_id": state.rep_id,
                    "limb_id": limb_id,
                    "duration_s": float(duration),
                    "hip_vertical_range": float(hip_vertical_range),
                    "knee_min_angle": knee_min_angle,
                    "elbow_min_angle": elbow_min_angle,
                    "torso_max_lean_deg": torso_max_lean_deg,
                    "left_right_asymmetry": 0.0,  # can compute later if needed
                    "movement_smoothness": 0.8,   # placeholder
                    "avg_confidence": avg_conf,
                    "exercise_hint": exercise_hint,
                }

                state.last_rep_end_time = end_time
                completed_reps.append(rep_summary)

    return completed_reps
