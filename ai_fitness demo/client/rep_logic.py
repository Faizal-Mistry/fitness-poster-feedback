# client/rep_logic.py 

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class SingleLimbState:
    state: str = "EXTENDED"          # "EXTENDED" or "FLEXED"
    rep_id: int = 0
    rep_start_time: Optional[float] = None

    hip_positions: list = field(default_factory=list)
    knee_angles: list = field(default_factory=list)
    elbow_angles: list = field(default_factory=list)
    torso_devs: list = field(default_factory=list)
    confidences: list = field(default_factory=list)

    last_rep_end_time: Optional[float] = None


@dataclass
class MultiRepState:
    """
    Holds a separate SingleLimbState for each logical limb:
      - "global"  : whole-body reps (squats, pushups, curls, lunges, mountain climber)
      - "left"    : left-side leg/arm  (for future alternating exercises)
      - "right"   : right-side leg/arm (for future alternating exercises)
    """
    limb_states: Dict[str, SingleLimbState] = field(default_factory=dict)


# ----------------- Per-exercise configuration -----------------
EXERCISE_CONFIG: Dict[str, Dict[str, Any]] = {
    "squat": {
        "limbs": ["global"],
        "primary_joint": "knee",
        "flexed_threshold": 35.0,
        "extended_threshold": 15.0,
        "min_rep_duration": 0.20,
        "min_rest_time": 0.08,
        "use_limb_delta": False,
        "limb_activation_delta": 0.0,
    },
    "pushup": {
        "limbs": ["global"],
        "primary_joint": "elbow",
        "flexed_threshold": 40.0,
        "extended_threshold": 20.0,
        "min_rep_duration": 0.18,
        "min_rest_time": 0.08,
        "use_limb_delta": False,
        "limb_activation_delta": 0.0,
    },
    "bicep_curl": {
        "limbs": ["global"],
        "primary_joint": "elbow",
        "flexed_threshold": 90.0,
        "extended_threshold": 20.0,
        "min_rep_duration": 0.15,
        "min_rest_time": 0.06,
        "use_limb_delta": False,
        "limb_activation_delta": 0.0,
    },
    "lunge": {
        "limbs": ["global"],
        "primary_joint": "knee",
        "flexed_threshold": 45.0,
        "extended_threshold": 20.0,
        "min_rep_duration": 0.22,
        "min_rest_time": 0.10,
        "use_limb_delta": False,
        "limb_activation_delta": 0.0,
    },
    "mountain_climber": {
        # Mountain climber: count 1 global rep per strong knee-drive cycle
        "limbs": ["global"],
        "primary_joint": "knee",
        # need decent knee drive to count rep
        "flexed_threshold": 50.0,       # stricter bend
        "extended_threshold": 25.0,     # back closer to straight
        # tuned for fast but not crazy-fast reps
        "min_rep_duration": 0.16,       # ignore ultra-tiny flicks
        "min_rest_time": 0.08,          # short cooldown
        # no per-leg gating in this version
        "use_limb_delta": False,
        "limb_activation_delta": 0.0,
    },
}

DEFAULT_CONFIG = {
    "limbs": ["global"],
    "primary_joint": "knee",
    "flexed_threshold": 35.0,
    "extended_threshold": 15.0,
    "min_rep_duration": 0.3,
    "min_rest_time": 0.2,
    "use_limb_delta": False,
    "limb_activation_delta": 0.0,
}


def get_exercise_config(exercise_name: Optional[str]) -> Dict[str, Any]:
    if exercise_name and exercise_name in EXERCISE_CONFIG:
        return EXERCISE_CONFIG[exercise_name]
    return DEFAULT_CONFIG


# -------------------------------------------------------------
# Main update function
# -------------------------------------------------------------

def update_multi_rep_state(
    multi_state: MultiRepState,
    features: Dict[str, Any],
    avg_confidence: float,
    exercise_hint: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generic multi-limb rep detection.

    - For exercises with limbs=["global"], we track a single state.
    - For limbs=["left","right"], we *can* track separate states for each side
      (useful for future alternating exercises if needed).
    - A rep is counted when state: EXTENDED -> FLEXED -> EXTENDED,
      with min_rep_duration and min_rest_time checks.
    """
    now = time.time()
    cfg = get_exercise_config(exercise_hint)

    limbs = cfg["limbs"]
    primary_joint = cfg["primary_joint"]
    flexed_threshold = cfg["flexed_threshold"]
    extended_threshold = cfg["extended_threshold"]
    min_rep_duration = cfg["min_rep_duration"]
    min_rest_time = cfg["min_rest_time"]
    use_limb_delta = cfg["use_limb_delta"]
    limb_activation_delta = cfg["limb_activation_delta"]

    # Ensure limb states exist
    for limb_id in limbs:
        if limb_id not in multi_state.limb_states:
            multi_state.limb_states[limb_id] = SingleLimbState()

    # Extract angles & positions from features
    left_knee_angle = features.get("left_knee_angle_frame", 180.0)
    right_knee_angle = features.get("right_knee_angle_frame", 180.0)
    left_elbow_angle = features.get("left_elbow_angle_frame", 180.0)
    right_elbow_angle = features.get("right_elbow_angle_frame", 180.0)

    knee_min_angle = features.get("knee_min_angle_frame", 180.0)
    elbow_min_angle = features.get("elbow_min_angle_frame", 180.0)
    torso_dev = features.get("torso_dev_frame", 0.0)
    center_hip_y = features.get("center_hip_y", 0.0)

    # Flex amounts: bigger = more bent
    left_knee_flex = 180.0 - left_knee_angle
    right_knee_flex = 180.0 - right_knee_angle
    left_elbow_flex = 180.0 - left_elbow_angle
    right_elbow_flex = 180.0 - right_elbow_angle

    completed_reps: List[Dict[str, Any]] = []

    def get_limb_joint_flex(limb: str):
        if primary_joint == "knee":
            if limb == "left":
                return left_knee_angle, left_knee_flex
            elif limb == "right":
                return right_knee_angle, right_knee_flex
            else:
                # "global" → use the most bent knee as driver
                return knee_min_angle, max(left_knee_flex, right_knee_flex)
        else:
            if limb == "left":
                return left_elbow_angle, left_elbow_flex
            elif limb == "right":
                return right_elbow_angle, right_elbow_flex
            else:
                # "global" → use the most bent elbow as driver
                return elbow_min_angle, max(left_elbow_flex, right_elbow_flex)

    for limb_id in limbs:
        state = multi_state.limb_states[limb_id]
        joint_angle, flex_amount = get_limb_joint_flex(limb_id)

        # ---------- Limb activation gating (currently only useful if we use left/right) ----------
        if use_limb_delta and limb_id in ("left", "right"):
            # Compare with the other side (for future alternating exercises)
            if limb_id == "left":
                _, other_flex = get_limb_joint_flex("right")
            else:
                _, other_flex = get_limb_joint_flex("left")

            # If THIS limb is not clearly more bent, treat it as almost straight
            # so it can't cross flexed_threshold and start a rep.
            if flex_amount < other_flex + limb_activation_delta:
                flex_amount = 0.0  # effectively "not active" this frame

        # -----------------------------
        # State machine per limb
        # -----------------------------
        if state.state == "EXTENDED":
            # Small cooldown after a rep
            if state.last_rep_end_time is not None:
                time_since_last = now - state.last_rep_end_time
                if time_since_last < min_rep_duration:
                    continue

            # Start rep when we bend enough
            if flex_amount > flexed_threshold:
                state.state = "FLEXED"
                state.rep_start_time = now

                state.hip_positions = [center_hip_y]
                state.knee_angles = [knee_min_angle]
                state.elbow_angles = [elbow_min_angle]
                state.torso_devs = [torso_dev]
                state.confidences = [avg_confidence]

        elif state.state == "FLEXED":
            # Accumulate during rep
            state.hip_positions.append(center_hip_y)
            state.knee_angles.append(knee_min_angle)
            state.elbow_angles.append(elbow_min_angle)
            state.torso_devs.append(torso_dev)
            state.confidences.append(avg_confidence)

            # Rep ends when nearly straight again
            if flex_amount < extended_threshold:
                state.state = "EXTENDED"
                end_time = now
                start_time = state.rep_start_time or end_time
                duration = end_time - start_time

                # Ignore ultra-short reps (noise)
                if duration < min_rep_duration:
                    state.rep_start_time = None
                    state.hip_positions.clear()
                    state.knee_angles.clear()
                    state.elbow_angles.clear()
                    state.torso_devs.clear()
                    state.confidences.clear()
                    state.last_rep_end_time = end_time
                    continue

                # Valid rep
                state.rep_id += 1

                hip_vertical_range = 0.0
                if state.hip_positions:
                    hip_vertical_range = (
                        max(state.hip_positions) - min(state.hip_positions)
                    )

                rep_summary = {
                    "rep_id": state.rep_id,
                    "limb_id": limb_id,
                    "duration_s": float(duration),
                    "hip_vertical_range": float(hip_vertical_range),
                    "knee_min_angle": float(min(state.knee_angles)) if state.knee_angles else 180.0,
                    "elbow_min_angle": float(min(state.elbow_angles)) if state.elbow_angles else 180.0,
                    "torso_max_lean_deg": float(max(state.torso_devs)) if state.torso_devs else 0.0,
                    "left_right_asymmetry": 0.0,
                    "movement_smoothness": 0.8,
                    "avg_confidence": float(np.mean(state.confidences)) if state.confidences else 0.0,
                    "exercise_hint": exercise_hint,
                }

                state.last_rep_end_time = end_time
                state.rep_start_time = None
                state.hip_positions.clear()
                state.knee_angles.clear()
                state.elbow_angles.clear()
                state.torso_devs.clear()
                state.confidences.clear()

                completed_reps.append(rep_summary)

    return completed_reps
