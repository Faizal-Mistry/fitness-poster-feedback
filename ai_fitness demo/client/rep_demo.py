# # client/rep_demo.py

import time
import cv2
import mediapipe as mp
import requests
import pyttsx3
from threading import Thread
from queue import Queue

from pose_utils import PoseEstimator
from rep_logic import MultiRepState, update_multi_rep_state, get_exercise_config

# MediaPipe drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------- Exercise options ----------
EXERCISE_OPTIONS = {
    "1": "squat",
    "2": "pushup",
    "3": "bicep_curl",
    "4": "lunge",
    "5": "mountain_climber",
}

# Backend (FastAPI) endpoint
BACKEND_URL = "http://127.0.0.1:8000/analyze_rep"

# ---------- Queue for background LLM calls ----------
rep_queue: Queue = Queue()     # completed reps to send to backend

# Global overlay from last LLM response
last_coaching_message: str = ""


def choose_exercise():
    print("Select exercise to track:")
    print("  1. Squat")
    print("  2. Pushup")
    print("  3. Bicep Curl")
    print("  4. Lunge")
    print("  5. Mountain Climber")
    choice = input("Enter 1, 2, 3, 4, or 5: ").strip()
    exercise = EXERCISE_OPTIONS.get(choice, "squat")
    print(f"\nYou selected: {exercise}\n")
    return exercise


# ---------- TTS helper (per-message thread) ----------

def speak_message(text: str):
    """
    Create a fresh pyttsx3 engine for THIS message only.
    Runs in its own thread so the camera loop never blocks.
    """
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


# ---------- Background LLM worker ----------

def llm_worker():
    """
    Runs in a background thread.
    Reads completed rep summaries from rep_queue,
    calls backend (FastAPI + LLM),
    updates last_coaching_message,
    and spawns a TTS thread for the message.
    """
    global last_coaching_message

    while True:
        rep_summary = rep_queue.get()
        try:
            # Small timeout: if LLM is slow, we just skip that message
            resp = requests.post(BACKEND_URL, json=rep_summary, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                msg = data.get("message", "")
                if msg:
                    last_coaching_message = msg

                    # 🔊 Speak in a separate short-lived thread
                    Thread(
                        target=speak_message,
                        args=(msg,),
                        daemon=True
                    ).start()
            else:
                print("LLM worker backend error:", resp.status_code, resp.text)
        except Exception as e:
            print("LLM worker exception:", e)
        finally:
            rep_queue.task_done()


def main():
    global last_coaching_message

    # 1) Choose exercise
    current_exercise = choose_exercise()

    # 2) Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 3) Init pose estimator & multi-rep state
    pose_estimator = PoseEstimator()
    multi_state = MultiRepState()

    # 4) Start background LLM worker
    Thread(target=llm_worker, daemon=True).start()

    # 5) 5-second countdown before tracking
    countdown_seconds = 5
    countdown_start = time.time()
    countdown_done = False

    print(f"Get into position... starting in {countdown_seconds} seconds.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # ---------- PHASE 1: Countdown ----------
        if not countdown_done:
            elapsed = time.time() - countdown_start
            remaining = countdown_seconds - int(elapsed)

            if remaining > 0:
                cv2.putText(display_frame,
                            f"Get ready: {remaining}",
                            (60, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 255),
                            3)
            else:
                countdown_done = True
                print("Go! Tracking reps now.")

            cv2.imshow("AI Fitness - Technometics Demo", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ---------- PHASE 2: Pose + rep tracking ----------
        features, landmarks = pose_estimator.process(frame)

        # Draw skeleton if we have landmarks
        if landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                ),
            )

        # If we have features → update rep logic
        if features is not None:
            avg_confidence = 0.95  # could compute from visibility if you want

            completed_reps = update_multi_rep_state(
                multi_state,
                features,
                avg_confidence,
                current_exercise
            )

            # ------- Overlays for exercise + reps -------
            cfg = get_exercise_config(current_exercise)
            limbs = cfg["limbs"]

            cv2.putText(display_frame,
                        f"Exercise: {current_exercise}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 255, 200),
                        2)

            if limbs == ["global"]:
                state = multi_state.limb_states.get("global")
                reps = state.rep_id if state else 0
                cv2.putText(display_frame,
                            f"Reps: {reps}",
                            (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2)
            else:
                left_state = multi_state.limb_states.get("left")
                right_state = multi_state.limb_states.get("right")
                left_reps = left_state.rep_id if left_state else 0
                right_reps = right_state.rep_id if right_state else 0
                total_reps = left_reps + right_reps

                cv2.putText(display_frame,
                            f"Left reps:  {left_reps}",
                            (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2)
                cv2.putText(display_frame,
                            f"Right reps: {right_reps}",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2)
                cv2.putText(display_frame,
                            f"Total reps: {total_reps}",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2)

            # ---------- Handle completed reps (non-blocking) ----------
            # completed_reps is a list of rep_summary dicts
            for rep_summary in completed_reps:
                rep_id = int(rep_summary.get("rep_id", 0))
                limb_id = rep_summary.get("limb_id", "global")

                print(f"=== REP COMPLETED (limb={limb_id}, rep_id={rep_id}) ===")
                print("Rep summary:", rep_summary)

                # Only send odd reps to backend + voice
                if rep_id % 2 == 1:
                    rep_queue.put(rep_summary)   # returns instantly

        # ---------- Coaching message overlay ----------
        if last_coaching_message:
            cv2.putText(display_frame,
                        last_coaching_message,
                        (20, display_frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2)

        cv2.imshow("AI Fitness - Technometics Demo", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

