import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def angle_between(a, b, c):
    """
    Returns the angle (in degrees) at point b formed by points a-b-c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    v1 = a - b
    v2 = c - b

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    return float(angle)

class PoseEstimator:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr):
        """
        Input: BGR frame from OpenCV.
        Output:
          - features: dict with numeric values for this frame
          - landmarks: pose_landmarks (for drawing), or None if not detected
        """
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, None

        lm = results.pose_landmarks.landmark

        def pt(idx):
            p = lm[idx]
            return (p.x * w, p.y * h), p.visibility

        # Hips / knees / ankles
        left_hip, _ = pt(23)
        right_hip, _ = pt(24)
        left_knee, _ = pt(25)
        right_knee, _ = pt(26)
        left_ankle, _ = pt(27)
        right_ankle, _ = pt(28)

        # Shoulders / elbows / wrists
        left_shoulder, _ = pt(11)
        right_shoulder, _ = pt(12)
        left_elbow, _ = pt(13)
        right_elbow, _ = pt(14)
        left_wrist, _ = pt(15)
        right_wrist, _ = pt(16)

        # Knee angles
        left_knee_angle = angle_between(left_hip, left_knee, left_ankle)
        right_knee_angle = angle_between(right_hip, right_knee, right_ankle)
        knee_min = min(left_knee_angle, right_knee_angle)

        # Elbow angles
        left_elbow_angle = angle_between(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = angle_between(right_shoulder, right_elbow, right_wrist)
        elbow_min = min(left_elbow_angle, right_elbow_angle)

        # Torso lean using shoulder-hip-ankle midline
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                        (left_shoulder[1] + right_shoulder[1]) / 2)
        mid_hip = ((left_hip[0] + right_hip[0]) / 2,
                   (left_hip[1] + right_hip[1]) / 2)
        mid_ankle = ((left_ankle[0] + right_ankle[0]) / 2,
                     (left_ankle[1] + right_ankle[1]) / 2)

        torso_angle = angle_between(mid_shoulder, mid_hip, mid_ankle)
        torso_dev = 180 - torso_angle  # deviation from vertical

        # Center hip Y for depth
        center_hip_y = mid_hip[1]

        features = {
            "knee_min_angle_frame": knee_min,
            "elbow_min_angle_frame": elbow_min,
            "torso_dev_frame": torso_dev,
            "center_hip_y": center_hip_y,
            "left_knee_angle_frame": float(left_knee_angle),
            "right_knee_angle_frame": float(right_knee_angle),
        }

        return features, results.pose_landmarks
