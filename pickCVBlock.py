#This code is a simplified implementation of a collaborative robotics system that detects plates and targets using computer vision, 
#and then commands a Dobot robotic arm to pick and place objects accordingly. The system operates in three phases: scanning for plates, 
#scanning for targets, and executing the pick/place operations. 
#Stability checks are implemented to ensure reliable detection before proceeding to the next phase.

# Note: there are parameters that are useful to the successful operation of the robot arm. Read through the code before running the program.

# How to use: 
# 1. Ensure you have the Dobot robotic arm set up and connected to your computer.
# 2. Place the plates (drop zones) and targets (red blocks) within the camera's
# field of view.
# 3. Run the script. The system will first scan for plates, then targets, and finally execute the pick/place operations based on the detected positions.
# 4. Monitor the console output and the video feed for feedback on the system's status and operations

#Other Useful Codes you can use:
#dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead): moves the robot to the specified (x, y, z) coordinates with a specified rotation for the end effector (rHead). Z_SAFE is a predefined constant that ensures the robot maintains a safe height to avoid collisions when moving horizontally.



import os
import sys

try:
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    pygame = None
    AUDIO_AVAILABLE = False

script_dir = os.path.dirname(os.path.abspath(__file__))

for candidate_dir in [script_dir, os.path.dirname(script_dir)]:
    lib_dir = os.path.join(candidate_dir, "lib")
    if os.path.isdir(lib_dir) and candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)
        break
for root, dirs, files in os.walk(script_dir):
    if "DobotDllType.py" in files:
        if root not in sys.path:
            sys.path.insert(0, root)
        break

import dobotArm
try:
    import lib.DobotDllType as dType
except ModuleNotFoundError:
    import DobotDllType as dType
import numpy as np
import cv2
import time
import math


"""CONSTANTS"""

Z_SAFE = 40 #what is the clearance distance for the robot arm to avoid collisions when moving horizontally?
Z_PICK = -25 #what is the  height for the robot claw to successfully pick up the target?
STABILITY_LIMIT = 60  #how many consecutive frames of stable detection before we "lock in" the positions and move to the next phase? (at 30fps, 60 frames is about 2 seconds)
PIXEL_TOLERANCE = 10  #object can move at most this # of pixels to be considered stationary

machine_state = "scanning plate" 

# --- INITIALIZATION FOR CAMERA TRANSFORMATION ---
# MAKE SURE THAT YOU HAVE RAN calibrateCamera.py FIRST TO GENERATE THE camera_params.npz FILE
api = dType.load()
cap = cv2.VideoCapture(0)
H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once
ret, frame = cap.read()
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

#
# Beep when robot action is resumed
#

RESUME_SOUND_PATH = os.path.join(script_dir, "resume_sound.mp3")

audio_ready = False

def setup_audio():
    global audio_ready

    if not AUDIO_AVAILABLE:
        print("[AUDIO] pygame is not installed. Run: python -m pip install pygame")
        return

    if not os.path.exists(RESUME_SOUND_PATH):
        print(f"[AUDIO] Resume sound not found: {RESUME_SOUND_PATH}")
        return

    try:
        pygame.mixer.init()
        audio_ready = True
        print("[AUDIO] Resume sound ready.")
    except Exception as exc:
        print(f"[AUDIO] Could not initialize audio: {exc}")


def play_resume_sound():
    if not audio_ready:
        return

    try:
        sound = pygame.mixer.Sound(RESUME_SOUND_PATH)
        sound.play()
    except Exception as exc:
        print(f"[AUDIO] Could not play resume sound: {exc}")

# -----------------------------
# HAND GESTURE DETECTION + EMERGENCY PAUSE/RESUME
# Uses the same MediaPipe/landmark rules as the standalone hand tracker.
# Rules:
#   Peace Sign  -> immediately request robot stop and pause program
#   Thumb Out   -> resume program after pause
#
# Important: this is a software stop, not a certified safety emergency stop.
# Keep a physical emergency stop available whenever the robot is powered.
# -----------------------------

import threading

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# Shared state between the robot thread and the hand-monitor thread.
pause_active = False
pause_state_text = ""
last_raw_gesture = "No hand detected"
program_running = True
emergency_stop_issued = False

# If the robot is interrupted during a command, repeat that same command after Thumb Out.
# This is usually safer than skipping to the next command after an interrupted move.
RETRY_INTERRUPTED_COMMAND_AFTER_RESUME = True

# Gesture safety filtering. Pause/resume are kept. Thumb Out must be held for several consecutive frames
# before resume. This prevents a fist, a partial hand, or a hand re-entering the
# camera view from accidentally resuming the robot.
PAUSE_REQUIRED_PEACE_FRAMES = 2
RESUME_REQUIRED_THUMB_FRAMES = 10
RESUME_DELAY_SECONDS = 3.0

pause_peace_counter = 0
resume_thumb_counter = 0
resume_delay_started_at = None

# Keep this False by default. Opening the gripper/stopping suction during an emergency
# could drop the object. Turn it on only if your use case requires release on stop.
RELEASE_END_EFFECTOR_ON_ESTOP = False

camera_lock = threading.Lock()
state_lock = threading.Lock()


def read_camera_frame():
    """Thread-safe camera read because the main loop and monitor thread both use cap."""
    with camera_lock:
        return cap.read()


def angle_between_points(a, b, c):
    ab = [a.x - b.x, a.y - b.y, a.z - b.z]
    cb = [c.x - b.x, c.y - b.y, c.z - b.z]

    dot_product = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]

    ab_length = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    cb_length = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)

    if ab_length == 0 or cb_length == 0:
        return 0

    cosine_angle = dot_product / (ab_length * cb_length)
    cosine_angle = max(-1, min(1, cosine_angle))
    return math.degrees(math.acos(cosine_angle))


def landmark_distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def palm_size_from_landmarks(lm):
    """Approximate hand scale so thresholds work at different camera distances."""
    # wrist to middle MCP is a stable palm-size reference
    return max(landmark_distance(lm[0], lm[9]), 1e-6)


def finger_is_extended(lm, mcp, pip, dip, tip):
    """
    Rotation-tolerant finger extension test for index/middle/ring/pinky.
    A finger must be fairly straight AND the fingertip must be farther from
    the wrist than the PIP joint. This avoids counting curled fingers as up.
    """
    pip_angle = angle_between_points(lm[mcp], lm[pip], lm[dip])
    dip_angle = angle_between_points(lm[pip], lm[dip], lm[tip])
    tip_dist = landmark_distance(lm[0], lm[tip])
    pip_dist = landmark_distance(lm[0], lm[pip])

    return (
        pip_angle > 150 and
        dip_angle > 145 and
        tip_dist > pip_dist * 1.06
    )


def thumb_is_extended_basic(hand_landmarks, hand_label=None):
    """
    General thumb-extension detector for display purposes.
    This is not enough by itself to resume the robot.
    """
    lm = hand_landmarks.landmark
    psize = palm_size_from_landmarks(lm)

    thumb_angle = angle_between_points(lm[2], lm[3], lm[4])
    thumb_tip_to_wrist = landmark_distance(lm[4], lm[0])
    thumb_ip_to_wrist = landmark_distance(lm[3], lm[0])
    thumb_tip_to_index_mcp = landmark_distance(lm[4], lm[5])
    thumb_ip_to_index_mcp = landmark_distance(lm[3], lm[5])

    return (
        thumb_angle > 150 and
        thumb_tip_to_wrist > thumb_ip_to_wrist * 1.03 and
        thumb_tip_to_index_mcp > thumb_ip_to_index_mcp * 1.08 and
        landmark_distance(lm[2], lm[4]) > psize * 0.30
    )


def is_thumb_out_control(hand_landmarks, hand_label=None):
    """
    Control-level Thumb Out detector used to RESUME the robot.

    This is stricter than normal thumb detection because a false positive here
    can resume the arm. It requires:
      - thumb is straight and separated from the palm/index side
      - thumb points mostly sideways
      - index/middle/ring/pinky are folded

    So a fist, a partial hand entering the frame, or a curled hand should not
    resume the robot.
    """
    lm = hand_landmarks.landmark
    psize = palm_size_from_landmarks(lm)

    # Other fingers must be folded for a resume command.
    index_up = finger_is_extended(lm, 5, 6, 7, 8)
    middle_up = finger_is_extended(lm, 9, 10, 11, 12)
    ring_up = finger_is_extended(lm, 13, 14, 15, 16)
    pinky_up = finger_is_extended(lm, 17, 18, 19, 20)
    if index_up or middle_up or ring_up or pinky_up:
        return False

    # Thumb geometry.
    thumb_angle = angle_between_points(lm[2], lm[3], lm[4])
    thumb_len = landmark_distance(lm[2], lm[4])
    thumb_tip_to_index_mcp = landmark_distance(lm[4], lm[5])
    thumb_ip_to_index_mcp = landmark_distance(lm[3], lm[5])
    thumb_tip_to_palm = landmark_distance(lm[4], lm[9])
    thumb_ip_to_palm = landmark_distance(lm[3], lm[9])

    # Direction of thumb from thumb MCP/base toward thumb tip.
    dx = lm[4].x - lm[2].x
    dy = lm[4].y - lm[2].y

    thumb_straight = thumb_angle > 152
    thumb_long_enough = thumb_len > psize * 0.33
    thumb_separated = (
        thumb_tip_to_index_mcp > thumb_ip_to_index_mcp * 1.10 and
        thumb_tip_to_palm > thumb_ip_to_palm * 1.06
    )
    thumb_sideways = abs(dx) > abs(dy) * 1.10 and abs(dx) > psize * 0.25

    return thumb_straight and thumb_long_enough and thumb_separated and thumb_sideways


def count_fingers_up(hand_landmarks, hand_label=None):
    """
    Refined landmark-based finger detection.
    Returns [thumb, index, middle, ring, pinky].
    The thumb value is for display/diagnostics; robot resume uses
    is_thumb_out_control(), which is stricter and state-gated.
    """
    lm = hand_landmarks.landmark

    thumb = 1 if thumb_is_extended_basic(hand_landmarks, hand_label) else 0
    index = 1 if finger_is_extended(lm, 5, 6, 7, 8) else 0
    middle = 1 if finger_is_extended(lm, 9, 10, 11, 12) else 0
    ring = 1 if finger_is_extended(lm, 13, 14, 15, 16) else 0
    pinky = 1 if finger_is_extended(lm, 17, 18, 19, 20) else 0

    return [thumb, index, middle, ring, pinky]


def get_pointing_direction(hand_landmarks):
    lm = hand_landmarks.landmark
    index_base = lm[5]
    index_tip = lm[8]

    dx = index_tip.x - index_base.x
    dy = index_tip.y - index_base.y

    if abs(dx) > abs(dy):
        if dx < -0.06:
            return "Pointing Left"
        if dx > 0.06:
            return "Pointing Right"
    else:
        if dy < -0.06:
            return "Pointing Up"
        if dy > 0.06:
            return "Pointing Down"

    return "Pointing"


def decode_gesture(fingers, hand_landmarks=None):
    """
    Human-readable gesture label.
    IMPORTANT: robot pause/resume does not rely only on this text. It uses
    decode_control_gesture() so accidental display labels cannot resume motion.
    """
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    if fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    if fingers == [0, 1, 0, 0, 0]:
        if hand_landmarks is not None:
            return get_pointing_direction(hand_landmarks)
        return "Index Finger"
    if fingers == [0, 0, 1, 0, 0]:
        return "Middle Finger"
    if fingers == [0, 0, 0, 1, 0]:
        return "Ring Finger"
    if fingers == [0, 0, 0, 0, 1]:
        return "Pinky Finger"
    if fingers in ([0, 1, 1, 0, 0], [1, 1, 1, 0, 0]):
        return "Peace Sign"
    if hand_landmarks is not None and is_thumb_out_control(hand_landmarks):
        return "Thumb Out"
    return f"{sum(fingers)} fingers extended: {fingers}"


def decode_control_gesture(hand_landmarks, hand_label=None):
    """
    Safety/control gesture decoder for robot pause/resume.
    Returns only command-level labels:
      Peace Sign, Thumb Out, or a non-command display gesture.
    """
    if hand_landmarks is None:
        return "No hand detected", [0, 0, 0, 0, 0]

    fingers = count_fingers_up(hand_landmarks, hand_label)

    # Pause command: index + middle extended, ring + pinky folded.
    # Thumb may be folded or slightly visible, so allow thumb 0 or 1.
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "Peace Sign", fingers

    # Resume command: stricter geometry and other fingers folded.
    if is_thumb_out_control(hand_landmarks, hand_label):
        return "Thumb Out", fingers

    return decode_gesture(fingers, hand_landmarks), fingers


def draw_top_text(frame, text, y_offset=40, font_scale=1.2, thickness=3, color=(0, 255, 255)):
    cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def detect_raw_hand_gesture(frame, display_frame=None, draw=True):
    """
    Detect gesture from the full camera frame. No action-area ROI is used.
    If draw=True, landmarks and finger states are drawn onto display_frame.
    """
    if display_frame is None:
        display_frame = frame

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return "No hand detected"

    raw_gesture = "No hand detected"

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        hand_label = handedness.classification[0].label
        raw_gesture, fingers = decode_control_gesture(hand_landmarks, hand_label)

        if draw:
            mp_draw.draw_landmarks(
                display_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            )
            cv2.putText(
                display_frame,
                f"Gesture: {raw_gesture} | Fingers: {fingers} | Hand: {hand_label}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    return raw_gesture


def _call_dobot_if_available(function_name):
    """Best-effort call into DobotDllType. Different Dobot libraries expose different names."""
    fn = getattr(dType, function_name, None)
    if callable(fn):
        try:
            fn(api)
            print(f"[DOBOT] Called {function_name}()")
            return True
        except Exception as exc:
            print(f"[DOBOT] {function_name}() failed: {exc}")
    return False


def robot_emergency_stop():
    """
    Best-effort immediate software stop.
    This is intended to interrupt queued Dobot motion while the main thread may be inside move_to_xyz().
    """
    print("[EMERGENCY] Peace Sign detected. Requesting robot stop NOW.")

    # Common Dobot queue-stop functions. These are best-effort because exact wrappers vary.
    stopped = False
    for name in [
        "SetQueuedCmdForceStopExec",
        "SetQueuedCmdStopExec",
        "SetPTPStop",
    ]:
        stopped = _call_dobot_if_available(name) or stopped

    # Clear queued future commands if supported.
    for name in ["SetQueuedCmdClear"]:
        _call_dobot_if_available(name)

    if RELEASE_END_EFFECTOR_ON_ESTOP:
        try:
            dobotArm.stop_pump(api)
            dobotArm.open_gripper(api)
            print("[EMERGENCY] End effector released because RELEASE_END_EFFECTOR_ON_ESTOP=True")
        except Exception as exc:
            print(f"[EMERGENCY] End effector release failed: {exc}")

    if not stopped:
        print("[WARNING] No Dobot force-stop function was available. Use the physical emergency stop if motion continues.")


def robot_resume_after_stop():
    """Best-effort resume of the Dobot command queue after Thumb Out."""
    print("[RESUME] Thumb Out detected. Resuming program.")

    play_resume_sound()

    for name in ["SetQueuedCmdStartExec"]:
        _call_dobot_if_available(name)


def update_pause_state_from_gesture(raw_gesture):
    global pause_active, pause_state_text, last_raw_gesture, emergency_stop_issued
    global pause_peace_counter, resume_thumb_counter, resume_delay_started_at

    last_raw_gesture = raw_gesture

    with state_lock:
        if pause_active:
            # While paused, ONLY a stable strict Thumb Out resumes the robot.
            # No hand, fist, open palm, and re-entering the frame all reset the counter.
            pause_peace_counter = 0

            if raw_gesture == "Thumb Out":
                resume_thumb_counter += 1

                # Step 1: wait until Thumb Out is stable for enough frames
                if resume_thumb_counter < RESUME_REQUIRED_THUMB_FRAMES:
                    resume_delay_started_at = None
                    pause_state_text = (
                        f"ROBOT PAUSED - hold Thumb Out "
                        f"({resume_thumb_counter}/{RESUME_REQUIRED_THUMB_FRAMES})"
                    )
                    return pause_state_text

                # Step 2: once Thumb Out is confirmed, start the 3-second delay
                if resume_delay_started_at is None:
                    resume_delay_started_at = time.time()

                elapsed = time.time() - resume_delay_started_at
                remaining = max(0.0, RESUME_DELAY_SECONDS - elapsed)

                if remaining <= 0:
                    pause_active = False
                    pause_state_text = ""
                    emergency_stop_issued = False
                    resume_thumb_counter = 0
                    resume_delay_started_at = None
                    robot_resume_after_stop()
                    return "Thumb Out confirmed - robot resumed"

                pause_state_text = f"THUMB OUT CONFIRMED - resuming in {remaining:.1f}s"
                return pause_state_text

            # If anything other than Thumb Out is seen, reset resume progress
            resume_thumb_counter = 0
            resume_delay_started_at = None
            pause_state_text = "ROBOT PAUSED - show a clear Thumb Out to resume"
            return pause_state_text
        # Not paused: only a stable Peace Sign pauses/stops the robot.
        resume_thumb_counter = 0

        if raw_gesture == "Peace Sign":
            pause_peace_counter += 1
            if pause_peace_counter >= PAUSE_REQUIRED_PEACE_FRAMES:
                pause_active = True
                pause_state_text = "ROBOT PAUSED - show a clear Thumb Out to resume"
                pause_peace_counter = 0
                if not emergency_stop_issued:
                    emergency_stop_issued = True
                    robot_emergency_stop()
                return pause_state_text

            return f"Peace Sign detected ({pause_peace_counter}/{PAUSE_REQUIRED_PEACE_FRAMES})"

        pause_peace_counter = 0
        pause_state_text = ""
        return raw_gesture


def detect_hand_gesture(frame):
    """Detect gesture on full frame and update pause/resume state."""
    raw_gesture = detect_raw_hand_gesture(frame, frame, draw=True)
    return update_pause_state_from_gesture(raw_gesture)


def hand_monitor_loop():
    """
    Background hand monitor.
    It checks the full camera frame continuously. Debouncing is handled in
    update_pause_state_from_gesture(), so a single false Thumb Out frame cannot resume.
    """
    global program_running

    while program_running:
        ret, frame = read_camera_frame()
        if not ret:
            time.sleep(0.02)
            continue

        try:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        except Exception:
            pass

        raw = detect_raw_hand_gesture(frame, draw=False)
        update_pause_state_from_gesture(raw)

        time.sleep(0.01)


def show_hand_status_frame(frame, gesture_text):
    if pause_active:
        draw_top_text(frame, "ROBOT PAUSED", y_offset=40, color=(0, 0, 255))
        draw_top_text(frame, "Show Thumb Out to resume", y_offset=85, font_scale=0.8, thickness=2, color=(0, 255, 255))
    else:
        draw_top_text(frame, gesture_text, y_offset=40, font_scale=1.0, thickness=2, color=(0, 255, 0))

    cv2.imshow("Detection", frame)
    cv2.waitKey(1)


def wait_for_resume():
    """Block the main program while paused. Only Thumb Out resumes."""
    while pause_active:
        ret, frame = read_camera_frame()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        gesture_text = detect_hand_gesture(display_frame)
        show_hand_status_frame(display_frame, gesture_text)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def guarded_robot_call(action_name, func, *args, **kwargs):
    """
    Run a robot action under the gesture emergency gate.
    If Peace Sign happens during this blocking call, the monitor thread requests a stop.
    After Thumb Out, this wrapper repeats the interrupted command by default.
    """
    while True:
        wait_for_resume()

        print(f"[ROBOT] Starting: {action_name}")
        result = func(*args, **kwargs)

        if pause_active:
            print(f"[ROBOT] {action_name} was interrupted. Waiting for Thumb Out...")
            wait_for_resume()
            if RETRY_INTERRUPTED_COMMAND_AFTER_RESUME:
                print(f"[ROBOT] Retrying interrupted command: {action_name}")
                continue

        return result



# State machine logic to control the flow of the program through the three phases: scanning for plates, scanning for targets, and executing pick/place operations.
# THIS STATE MACHINE IS TOO SIMPLE. Can you think of logics that should change the robot's sequnece of actions?
# Ex: what if the robot fails to pick up a target? should it retry? should it go back to scanning for targets in case the target was moved? what if a new plate is added during the pick/place phase?
# What if a human's hand is in sight during pick/place phase? (safety first!)

def next_state():
    global machine_state
    if machine_state == "scanning plate":
        machine_state = "scanning target"
    elif machine_state == "scanning target":
        machine_state = "pick place"
    elif machine_state == "pick place":
        machine_state = "scanning plate"
    else:
        machine_state = "scanning plate"



# ---------------------------------------------------------
# PHASE 1: DETECT Part Drop Zones (Plates)
# this script assumes a metallic circular plate as the drop zone, but you can modify the detection logic to fit your specific use case.
# ---------------------------------------------------------
def phase_detect_plates():
    print("\n[PHASE 1] Scanning for drop zones. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = read_camera_frame()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=35, minRadius=25, maxRadius=55)

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :] :
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        # --- AUTO-LOCK LOGIC ---
        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        gesture_text = detect_hand_gesture(display_frame)
        if pause_active:
            stability_counter = 0
            show_hand_status_frame(display_frame, gesture_text)
            continue

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list
  
 

# ---------------------------------------------------------
# PHASE 2: DETECT Red velcros to pick up (Red Blocks)
# this script assumes the targets to be picked up are red blocks
# be aware your target maynot be red, and they may not be rectangular! You will need to modify the detection logic to fit your specific use case.
# ---------------------------------------------------------
def phase_detect_targets():
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = read_camera_frame()
        if not ret: continue
        
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        # Create a display copy so drawings don't affect next frame's HSV detection
        display_frame = frame.copy()
        
        # Red Tag Logic
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 800:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    rx, ry = pixel_to_robot(cx, cy, H_matrix)
                    current_list.append((rx, ry))
                    # Draw on display_frame only
                    cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
                    
        # --- STABILITY LOGIC ---
        if len(current_list) != 0:
            if len(current_list) > 0 and len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count = len(current_list)

        gesture_text = detect_hand_gesture(display_frame)
        if pause_active:
            stability_counter = 0
            show_hand_status_frame(display_frame, gesture_text)
            continue

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)
        
        # --- EXIT CONDITION ---
        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_list)} targets.")
            #cv2.waitKey(500) # Brief pause so you can see the 100%
    
            return current_list


# ---------------------------------------------------------
# PHASE 3: PICK/PLACE LOOP
# This function assumes 1 drop zone only has 1 part, and executes the pick/place operations in batches.
# if you are picking up rigid car parts, would you still be able to move directly to the object and to the drop zone? 
# Do you need collision avoidance? Think about if the robot gripper accidentally hits the plate or other parts on the way to the target, what would happen? How would you modify the robot's movement logic to avoid collisions?
# ---------------------------------------------------------
def phase_execute_batch(api, pick_list, drop_list):
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("missing targets, aborting")
        return False
    
    # Match 1 part to 1 drop zone (uses the smaller count)
    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 3] Executing batch of {batch_size} operations.")

    for i in range(batch_size):
        wait_for_resume()
        pick_x, pick_y = pick_list[i]
        drop_x, drop_y = drop_list[i]

        print(f"Task {i+1}: Moving {pick_x, pick_y} to {drop_x, drop_y}")

        # --- PICK SEQUENCE ---
        wait_for_resume()
        guarded_robot_call("move above target", dobotArm.move_to_xyz, api, pick_x, pick_y, Z_SAFE)
        wait_for_resume()
        guarded_robot_call("move down to pick", dobotArm.move_to_xyz, api, pick_x, pick_y, Z_PICK)
        wait_for_resume()
        guarded_robot_call("close gripper", dobotArm.close_gripper, api)
        wait_for_resume()
        guarded_robot_call("move above target", dobotArm.move_to_xyz, api, pick_x, pick_y, Z_SAFE)

        # --- PLACE SEQUENCE ---
        wait_for_resume()
        guarded_robot_call("move above drop zone", dobotArm.move_to_xyz, api, drop_x, drop_y, Z_SAFE)
        wait_for_resume()
        guarded_robot_call("open gripper", dobotArm.open_gripper, api)
        wait_for_resume()
        guarded_robot_call("stop pump", dobotArm.stop_pump, api)
        wait_for_resume()
        guarded_robot_call("move above drop zone", dobotArm.move_to_xyz, api, drop_x, drop_y, Z_SAFE)

    # irl, it is ok for 1 dish to contain multiple parts
    # if len(pick_list) > len(drop_list):
    #     for i in range(len(pick_list)):
    #         pick_x, pick_y = pick_list[i]
    #         drop_x, drop_y = drop_list[0]
    #         # --- PICK SEQUENCE ---
    #         guarded_robot_call("move above target", dobotArm.move_to_xyz, api, pick_x, pick_y, Z_SAFE)
    #         guarded_robot_call("move down to pick", dobotArm.move_to_xyz, api, pick_x, pick_y, Z_PICK)
    #         guarded_robot_call("close gripper", dobotArm.close_gripper, api)
    #         guarded_robot_call("move above target", dobotArm.move_to_xyz, api, pick_x, pick_y, Z_SAFE)

    #     # --- PLACE SEQUENCE ---
    #         guarded_robot_call("move above drop zone", dobotArm.move_to_xyz, api, drop_x, drop_y, Z_SAFE)
    #         guarded_robot_call("open gripper", dobotArm.open_gripper, api)
    #         guarded_robot_call("stop pump", dobotArm.stop_pump, api)
    #         guarded_robot_call("move above drop zone", dobotArm.move_to_xyz, api, drop_x, drop_y, Z_SAFE)

    print("\nBatch Complete.")
    return True
 

setup_audio()

# Start background hand monitor before robot begins moving.
hand_monitor_thread = threading.Thread(target=hand_monitor_loop, daemon=True)
hand_monitor_thread.start()

# ---------------------------------------------------------
# MAIN EXECUTION
# contains an oversimplified state machine that runs the three phases sequentially. You can modify the logic to fit your specific use case.
# ---------------------------------------------------------
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

while machine_state == "scanning plate":
    drop_zone = phase_detect_plates()
    if drop_zone is not None:
        next_state()


while machine_state == "scanning target":
    pick_target = phase_detect_targets()
    if pick_target is not None:
        next_state()


while machine_state == "pick place":
    completed = phase_execute_batch(api, pick_target, drop_zone)
    if completed:
        next_state()
    else: break


cap.release()
cv2.destroyAllWindows()