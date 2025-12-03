import time
import cv2
import mediapipe as mp
import requests
import pyttsx3
from threading import Thread

from pose_utils import PoseEstimator
from rep_logic import RepState, update_rep_state

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Exercise options (now 5)
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
    exercise = EXERCISE_OPTIONS.get(choice, "squat")
    print(f"\nYou selected: {exercise}\n")
    return exercise


def speak_message(text: str):
    """Run TTS in a background thread with a fresh engine."""
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
        print("Error: Could not open camera/video.")
        return

    pose_estimator = PoseEstimator()
    rep_state = RepState()

    countdown_seconds = 5
    countdown_start_time = time.time()
    countdown_done = False

    last_coaching_message = ""   # shown ONLY on odd reps
    tts_thread = None

    print("Get into position... starting in 5 seconds.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # ---------------- COUNTDOWN ---------------- #
        if not countdown_done:
            elapsed = time.time() - countdown_start_time
            remaining = int(countdown_seconds - elapsed) + 1
            if remaining > 0:
                cv2.putText(display_frame, f"Get into position: {remaining}", (60, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                countdown_done = True
                print("Go! Tracking reps now.")

            cv2.imshow("AI Fitness Rep Demo - Press Q to quit", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ---------------- REP TRACKING ---------------- #
        features, landmarks = pose_estimator.process(frame)

        # Draw skeleton
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

        if features is not None:
            avg_confidence = 0.95  # could be computed from visibility later

            rep_completed, rep_summary = update_rep_state(
                rep_state,
                features,
                avg_confidence,
                exercise_hint=current_exercise,
            )

            # Basic overlays (no per-rep angles to keep it clean)
            cv2.putText(display_frame, f"Exercise: {current_exercise}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            cv2.putText(display_frame, f"Reps: {rep_state.rep_id}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show coaching text ONLY if it's from the last odd rep
            if last_coaching_message:
                cv2.putText(display_frame, last_coaching_message,
                            (20, display_frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if rep_completed and rep_summary:
                print("=== REP COMPLETED ===")
                print("Rep summary:", rep_summary)

                try:
                    resp = requests.post(BACKEND_URL, json=rep_summary, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        message = data.get("message", "")
                        print("Coaching response:", data)

                        # odd reps only: 1,3,5,7...
                        if rep_state.rep_id % 2 == 1 and message:
                            last_coaching_message = message

                            # voice also only on odd reps
                            if tts_thread is None or not tts_thread.is_alive():
                                tts_thread = Thread(
                                    target=speak_message,
                                    args=(message,),
                                    daemon=True,
                                )
                                tts_thread.start()
                        else:
                            # even reps: clear message (no text, no voice)
                            last_coaching_message = ""

                    else:
                        print("Backend error:", resp.status_code, resp.text)
                        # you can optionally show "Backend error" if you want
                except Exception as e:
                    print("Error calling backend:", e)

        cv2.imshow("AI Fitness Rep Demo - Press Q to quit", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
