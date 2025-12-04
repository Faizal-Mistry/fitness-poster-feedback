# client/rep_demo_multi.py

import time
import cv2
import mediapipe as mp
import requests
import pyttsx3
from threading import Thread

from pose_utils import PoseEstimator
from rep_logic import MultiRepState, update_multi_rep_state, get_exercise_config

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

EXERCISE_OPTIONS = {
    "1": "squat",
    "2": "pushup",
    "3": "bicep_curl",
    "4": "lunge",
    "5": "mountain_climber",
}

BACKEND_URL = "http://127.0.0.1:8000/analyze_rep"


def choose_exercise():
    print("Select exercise to track:")
    print("  1. Squat")
    print("  2. Pushup")
    print("  3. Bicep Curl")
    print("  4. Lunge")
    print("  5. Mountain Climber")
    choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

    return EXERCISE_OPTIONS.get(choice, "squat")


def speak_message(text):
    if not text:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("TTS error:", e)


def main():
    current_exercise = choose_exercise()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    pose_estimator = PoseEstimator()
    multi_state = MultiRepState()

    countdown_seconds = 5
    countdown_start_time = time.time()
    countdown_done = False

    last_coaching_message = ""
    tts_thread = None

    print(f"Get into position... starting in {countdown_seconds} seconds.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # COUNTDOWN PHASE
        if not countdown_done:
            remaining = countdown_seconds - int(time.time() - countdown_start_time)
            if remaining > 0:
                cv2.putText(display_frame, f"Get ready: {remaining}",
                            (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                countdown_done = True
                print("Go! Tracking reps now.")

            cv2.imshow("AI Fitness", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # POSE + REP LOGIC
        features, landmarks = pose_estimator.process(frame)

        if landmarks:
            mp_drawing.draw_landmarks(
                display_frame, landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        if features is not None:

            avg_confidence = 0.95
            completed_reps = update_multi_rep_state(
                multi_state, features, avg_confidence, current_exercise
            )

            cfg = get_exercise_config(current_exercise)
            limbs = cfg["limbs"]

            cv2.putText(display_frame, f"Exercise: {current_exercise}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

            # REP DISPLAY
            if limbs == ["global"]:
                state = multi_state.limb_states.get("global")
                reps = state.rep_id if state else 0
                cv2.putText(display_frame, f"Reps: {reps}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                left = multi_state.limb_states.get("left").rep_id if "left" in multi_state.limb_states else 0
                right = multi_state.limb_states.get("right").rep_id if "right" in multi_state.limb_states else 0
                total = left + right

                cv2.putText(display_frame, f"Left reps:  {left}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Right reps: {right}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Total reps: {total}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # COACHING PHASE — ONLY ODD REPS GIVE TEXT + VOICE
            for rep_summary in completed_reps:
                rep_id = int(rep_summary.get("rep_id", 0))

                # Skip even reps fully
                if rep_id % 2 == 0:
                    print(f"Rep {rep_id} (even) → skip coaching.")
                    continue

                print("=== ODD REP COACHING ===")
                print("Rep summary:", rep_summary)

                try:
                    resp = requests.post(BACKEND_URL, json=rep_summary, timeout=8)
                    if resp.status_code == 200:
                        data = resp.json()
                        last_coaching_message = data.get("message", "")
                        print("Coaching:", last_coaching_message)

                        # TTS on a background thread (never blocking)
                        if last_coaching_message:
                            if tts_thread is None or not tts_thread.is_alive():
                                tts_thread = Thread(target=speak_message,
                                                    args=(last_coaching_message,),
                                                    daemon=True)
                                tts_thread.start()
                except Exception as e:
                    print("Backend error:", e)

        # SHOW LAST COACHING MESSAGE
        if last_coaching_message:
            cv2.putText(display_frame, last_coaching_message,
                        (20, display_frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("AI Fitness", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
