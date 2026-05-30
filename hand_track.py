import cv2
import time
import math
import numpy as np


# -----------------------------
# SETUP OPENCV HAND TRACKING
# Hand detection via skin contour analysis (no MediaPipe dependency)
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


def detect_hand_gesture_cv(frame):
    """
    Detect hand gestures using OpenCV contour analysis.
    Returns: gesture_text, fingers_text, hand_label, contour (or None)
    """
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
        hand_label = "Right"  # Default; OpenCV doesn't distinguish left/right
    
    return gesture_text, fingers_text, hand_label, contour


# Legacy function names for backward compatibility (kept but simplified)
def angle_between_points(a, b, c):
    """Legacy function stub - not used in OpenCV mode."""
    return 0


def distance(a, b):
    """Legacy function stub - not used in OpenCV mode."""
    return 0


def count_fingers_up(hand_landmarks, hand_label=None):
    """Legacy function stub - not used in OpenCV mode."""
    return [0, 0, 0, 0, 0]


def get_pointing_direction(hand_landmarks):
    """Legacy function stub - not used in OpenCV mode."""
    return "Pointing"


def decode_gesture(fingers, hand_landmarks=None):
    """Legacy function stub - not used in OpenCV mode."""
    return "Hand detected"

# -----------------------------
# START CAMERA
# -----------------------------

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

previous_time = 0

print("Camera started. Press 'q' to quit.")

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

    # Flip frame horizontally so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # Detect hand gesture using OpenCV
    gesture_text, fingers_text, hand_label, contour = detect_hand_gesture_cv(frame)
    
    # Draw contour and hull if hand detected
    if contour is not None:
        hull = cv2.convexHull(contour)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)

    # Pause state handling: open palm pauses robot actions until thumb out is seen
    if pause_active:
        if gesture_text == "Thumb Out":
            pause_active = False
            pause_state_text = "Robot actions resumed"
            gesture_text = "Thumb Out detected - resuming robot"
        else:
            pause_state_text = "Robot actions paused"
            gesture_text = "Robot paused - show Thumb Out"
    elif gesture_text == "Open Palm":
        pause_active = True
        pause_state_text = "Robot actions paused"
        gesture_text = "Open Palm detected - robot paused"

    # -----------------------------
    # FPS CALCULATION
    # -----------------------------

    current_time = time.time()
    fps = 1 / (current_time - previous_time) if previous_time != 0 else 0
    previous_time = current_time

    # -----------------------------
    # DISPLAY TEXT ON SCREEN
    # -----------------------------

    cv2.putText(
        frame,
        gesture_text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        pause_state_text,
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        fingers_text,
        (30, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (30, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press q to quit",
        (30, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show camera window
    cv2.imshow("Hand Tracker", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------

cap.release()
cv2.destroyAllWindows()