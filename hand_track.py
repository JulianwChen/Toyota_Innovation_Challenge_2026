import cv2
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> a6356bf7ecba3982c5fa85f90dac4eddd5af47a2
import time
import math
import numpy as np

USE_MEDIAPIPE = False
mp = None
mp_hands = None
mp_draw = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        USE_MEDIAPIPE = True
    else:
        import importlib
        try:
            mp_py = importlib.import_module('mediapipe.python')
            if hasattr(mp_py, 'solutions'):
                mp_hands = mp_py.solutions.hands
                mp_draw = mp_py.solutions.drawing_utils
                USE_MEDIAPIPE = True
            else:
                from mediapipe.python import solutions
                mp_hands = solutions.hands
                mp_draw = solutions.drawing_utils
                USE_MEDIAPIPE = True
        except (ImportError, ModuleNotFoundError):
            pass
except (AttributeError, ImportError, ModuleNotFoundError):
    mp = None
    mp_hands = None
    mp_draw = None

if not USE_MEDIAPIPE:
    raise ImportError('MediaPipe import failed: mediapipe.solutions is unavailable. Install a supported mediapipe package and retry.')


# -----------------------------
<<<<<<< HEAD
# SETUP MEDIAPIPE HAND TRACKING
# -----------------------------

hands = mp_hands.Hands(
    static_image_mode=False,        # False = video/live camera mode
    max_num_hands=1,                # Track one hand for now
    min_detection_confidence=0.4,   # How confident before detecting hand
    min_tracking_confidence=0.4     # How confident before tracking hand
)


# -----------------------------
# FINGER DETECTION FUNCTION
=======
# SETUP OPENCV HAND TRACKING
# Hand detection via skin contour analysis (no MediaPipe dependency)
>>>>>>> a6356bf7ecba3982c5fa85f90dac4eddd5af47a2
# -----------------------------


# Hand detection via skin mask (HSV + YCrCb)
def get_skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([25, 150, 255], dtype=np.uint8)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = [c for c in contours if cv2.contourArea(c) > 2000]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def count_hand_defects(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        if b == 0 or c == 0:
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c)))))
        if angle < 90 and d > 2500:
            count += 1
    return count

    cosine_angle = dot_product / (ab_length * cb_length)
    cosine_angle = max(-1, min(1, cosine_angle))


# Legacy function names for backward compatibility (kept but simplified)
def angle_between_points(a, b, c):
    """Legacy function stub - not used in OpenCV mode."""
    return 0


def distance(a, b):
    """Legacy function stub - not used in OpenCV mode."""
    return 0


def palm_size_from_landmarks(lm):
    """Approximate hand scale so thresholds work at different camera distances."""
    return max(distance(lm[0], lm[9]), 1e-6)


def finger_is_extended(lm, mcp, pip, dip, tip):
    """
    Rotation-tolerant finger extension test for index/middle/ring/pinky.
    """
    pip_angle = angle_between_points(lm[mcp], lm[pip], lm[dip])
    dip_angle = angle_between_points(lm[pip], lm[dip], lm[tip])
    tip_dist = distance(lm[0], lm[tip])
    pip_dist = distance(lm[0], lm[pip])

    return (
        pip_angle > 150 and
        dip_angle > 145 and
        tip_dist > pip_dist * 1.06
    )


def thumb_is_extended_basic(hand_landmarks, hand_label=None):
    """
    General thumb-extension detector for display purposes.
    This is intentionally less important than is_thumb_out_control().
    """
    lm = hand_landmarks.landmark
    psize = palm_size_from_landmarks(lm)

    thumb_angle = angle_between_points(lm[2], lm[3], lm[4])
    thumb_tip_to_wrist = distance(lm[4], lm[0])
    thumb_ip_to_wrist = distance(lm[3], lm[0])
    thumb_tip_to_index_mcp = distance(lm[4], lm[5])
    thumb_ip_to_index_mcp = distance(lm[3], lm[5])

    return (
        thumb_angle > 150 and
        thumb_tip_to_wrist > thumb_ip_to_wrist * 1.03 and
        thumb_tip_to_index_mcp > thumb_ip_to_index_mcp * 1.08 and
        distance(lm[2], lm[4]) > psize * 0.30
    )


def is_thumb_out_control(hand_landmarks, hand_label=None):
    """
    Control-level Thumb Out detector.
    Use this for resume, not just fingers == [1,0,0,0,0].
    """
    lm = hand_landmarks.landmark
    psize = palm_size_from_landmarks(lm)

    index_up = finger_is_extended(lm, 5, 6, 7, 8)
    middle_up = finger_is_extended(lm, 9, 10, 11, 12)
    ring_up = finger_is_extended(lm, 13, 14, 15, 16)
    pinky_up = finger_is_extended(lm, 17, 18, 19, 20)
    if index_up or middle_up or ring_up or pinky_up:
        return False

    thumb_angle = angle_between_points(lm[2], lm[3], lm[4])
    thumb_len = distance(lm[2], lm[4])
    thumb_tip_to_index_mcp = distance(lm[4], lm[5])
    thumb_ip_to_index_mcp = distance(lm[3], lm[5])
    thumb_tip_to_palm = distance(lm[4], lm[9])
    thumb_ip_to_palm = distance(lm[3], lm[9])

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
<<<<<<< HEAD
    """
    Refined landmark-based finger detection.
    Returns [thumb, index, middle, ring, pinky].
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


# -----------------------------
# BASIC GESTURE DECODER
# -----------------------------

def decode_gesture(fingers, hand_landmarks=None):
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

    # Important: Thumb Out is only returned if the stricter thumb-out geometry passes.
    if hand_landmarks is not None and is_thumb_out_control(hand_landmarks):
        return "Thumb Out"

    return f"{sum(fingers)} fingers extended"


def decode_control_gesture(hand_landmarks, hand_label=None):
    """
    Gesture label intended for robot control.
    Pause = Peace Sign. Resume = Thumb Out.
    """
    if hand_landmarks is None:
        return "No hand detected", [0, 0, 0, 0, 0]

    fingers = count_fingers_up(hand_landmarks, hand_label)

    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "Peace Sign", fingers

    if is_thumb_out_control(hand_landmarks, hand_label):
        return "Thumb Out", fingers

    return decode_gesture(fingers, hand_landmarks), fingers


def get_skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
    upper_hsv = np.array([25, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = [c for c in contours if cv2.contourArea(c) > 2000]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def count_hand_defects(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        if b == 0 or c == 0:
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c)))))
        if angle < 90 and d > 2500:
            count += 1
    return count


def detect_hand_gesture_cv(frame):
    gesture_text = "No hand detected"
    fingers_text = ""
    hand_label = ""
    contour = None

    mask = get_skin_mask(frame)
    largest_contour = find_largest_contour(mask)

    if largest_contour is not None:
        contour = largest_contour
        defects = count_hand_defects(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        area = cv2.contourArea(largest_contour)

        if defects >= 4 and area > 7000:
            gesture_text = "Open Palm"
        elif defects <= 1 and aspect_ratio > 1.3 and area > 3500:
            gesture_text = "Thumb Out"
        else:
            gesture_text = "Hand detected"

        fingers_text = f"Defects: {defects} | Area: {int(area)} | Ratio: {aspect_ratio:.2f}"
        hand_label = "Right"

    return gesture_text, fingers_text, hand_label, contour

=======
    """Legacy function stub - not used in OpenCV mode."""
    return [0, 0, 0, 0, 0]


def get_pointing_direction(hand_landmarks):
    """Legacy function stub - not used in OpenCV mode."""
    return "Pointing"


def decode_gesture(fingers, hand_landmarks=None):
    """Legacy function stub - not used in OpenCV mode."""
    return "Hand detected"
>>>>>>> a6356bf7ecba3982c5fa85f90dac4eddd5af47a2

# -----------------------------
# START CAMERA
# -----------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

previous_time = 0

print("Camera started. Press 'q' to quit.")


# -----------------------------
# MAIN LOOP
# -----------------------------

while False:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    gesture_text = "No hand detected"
    fingers_text = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = count_fingers_up(hand_landmarks, hand_label)
            gesture_text = decode_gesture(fingers, hand_landmarks)
            fingers_text = f"Fingers: {fingers} | Hand: {hand_label}"

    current_time = time.time()
    fps = 1 / (current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time

    cv2.putText(frame, gesture_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(frame, fingers_text, (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press q to quit", (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Skip the first dummy loop cleanup and continue to the pause/resume loop
pause_active = False
pause_state_text = "Robot actions enabled"

# -----------------------------
# MAIN LOOP
# -----------------------------

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

<<<<<<< HEAD
    gesture_text = "No hand detected"
    fingers_text = ""

    if USE_MEDIAPIPE:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                )
                fingers = count_fingers_up(hand_landmarks, hand_label)
                gesture_text = decode_gesture(fingers, hand_landmarks)
                fingers_text = f"Fingers: {fingers} | Hand: {hand_label}"
    else:
        gesture_text, fingers_text, hand_label, contour = detect_hand_gesture_cv(frame)
        if contour is not None:
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)

=======
    # Detect hand gesture using OpenCV
    gesture_text, fingers_text, hand_label, contour = detect_hand_gesture_cv(frame)
    
    # Draw contour and hull if hand detected
    if contour is not None:
        hull = cv2.convexHull(contour)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)

    # Pause state handling: open palm pauses robot actions until thumb out is seen
>>>>>>> a6356bf7ecba3982c5fa85f90dac4eddd5af47a2
    if pause_active:
        if gesture_text == "Thumb Out":
            pause_active = False
            pause_state_text = "Robot actions resumed"
            gesture_text = "Thumb Out detected - resuming robot"
        else:
            pause_state_text = "Robot actions paused"
            gesture_text = "Robot paused"
    elif gesture_text == "Peace Sign":
        pause_active = True
        pause_state_text = "Robot actions paused"
        gesture_text = "Peace Sign detected - robot paused"
    else:
        pause_state_text = ""

    current_time = time.time()
    fps = 1 / (current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time

    cv2.putText(frame, gesture_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(frame, pause_state_text, (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, fingers_text, (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press q to quit", (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
