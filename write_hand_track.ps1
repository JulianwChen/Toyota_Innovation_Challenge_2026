$path = "C:\Users\Julia\OneDrive\Desktop\Toyota Innovation Challenge '26\hand_track.py"
$content = @'
import cv2
import mediapipe as mp
import time
import math


# -----------------------------
# SETUP MEDIAPIPE HAND TRACKING
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,        # False = video/live camera mode
    max_num_hands=1,                # Track one hand for now
    min_detection_confidence=0.4,   # How confident before detecting hand
    min_tracking_confidence=0.4     # How confident before tracking hand
)


# -----------------------------
# FINGER DETECTION FUNCTION
# -----------------------------


def angle_between_points(a, b, c):
    """
    Returns the angle at point b formed by points a-b-c.
    Example: angle at the middle finger joint.
    """

    ab = [a.x - b.x, a.y - b.y, a.z - b.z]
    cb = [c.x - b.x, c.y - b.y, c.z - b.z]

    dot_product = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]

    ab_length = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
    cb_length = math.sqrt(cb[0] ** 2 + cb[1] ** 2 + cb[2] ** 2)

    if ab_length == 0 or cb_length == 0:
        return 0

    cosine_angle = dot_product / (ab_length * cb_length)

    # Prevent math domain errors
    cosine_angle = max(-1, min(1, cosine_angle))

    angle = math.degrees(math.acos(cosine_angle))
    return angle


def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def count_fingers_up(hand_landmarks, hand_label=None):
    """
    More accurate finger detection using joint angles.

    Returns:
    [thumb, index, middle, ring, pinky]
    """

    lm = hand_landmarks.landmark

    fingers = []

    # -----------------------------
    # THUMB
    # -----------------------------
    # Thumb is special, so we use thumb angle + distance.
    thumb_angle = angle_between_points(lm[2], lm[3], lm[4])
    thumb_tip_to_index_base = distance(lm[4], lm[5])
    thumb_base_to_index_base = distance(lm[2], lm[5])

    thumb_extended = thumb_angle > 145 and thumb_tip_to_index_base > thumb_base_to_index_base * 0.7

    fingers.append(1 if thumb_extended else 0)

    # -----------------------------
    # INDEX, MIDDLE, RING, PINKY
    # -----------------------------
    # Each finger has:
    # MCP = base knuckle
    # PIP = middle joint
    # DIP = upper joint
    # TIP = fingertip

    finger_joints = [
        (5, 6, 7, 8),      # index
        (9, 10, 11, 12),   # middle
        (13, 14, 15, 16),  # ring
        (17, 18, 19, 20)   # pinky
    ]

    for mcp, pip, dip, tip in finger_joints:
        # Angle at PIP joint
        pip_angle = angle_between_points(lm[mcp], lm[pip], lm[dip])

        # Angle at DIP joint
        dip_angle = angle_between_points(lm[pip], lm[dip], lm[tip])

        # Fingertip should also be farther from wrist than the middle joint
        tip_dist = distance(lm[0], lm[tip])
        pip_dist = distance(lm[0], lm[pip])

        # A finger is extended if both joints are fairly straight
        # and the fingertip is clearly out from the palm.
        is_extended = (
            pip_angle > 150 and
            dip_angle > 150 and
            tip_dist > pip_dist * 1.05
        )

        fingers.append(1 if is_extended else 0)

    return fingers


def get_pointing_direction(hand_landmarks):
    """
    Detects whether the index finger is pointing left, right, up, or down.
    Uses index finger base landmark 5 and index fingertip landmark 8.
    """

    lm = hand_landmarks.landmark

    index_base = lm[5]
    index_tip = lm[8]

    dx = index_tip.x - index_base.x
    dy = index_tip.y - index_base.y

    # If horizontal movement is stronger than vertical movement
    if abs(dx) > abs(dy):
        if dx < -0.06:
            return "Pointing Left"
        elif dx > 0.06:
            return "Pointing Right"

    # If vertical movement is stronger than horizontal movement
    else:
        if dy < -0.06:
            return "Pointing Up"
        elif dy > 0.06:
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
        return "Index Pointing"

    if fingers == [0, 1, 1, 0, 0]:
        return "Peace Sign"

    if fingers == [1, 0, 0, 0, 0]:
        return "Thumb Out"

    return f"{sum(fingers)} fingers extended: {fingers}"

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

    # Convert BGR to RGB because MediaPipe uses RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Default values
    gesture_text = "No hand detected"
    fingers_text = ""

    # If hand is detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # Get whether MediaPipe thinks this is left or right hand
            hand_label = handedness.classification[0].label

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Count fingers
            fingers = count_fingers_up(hand_landmarks, hand_label)

            # Decode gesture
            gesture_text = decode_gesture(fingers, hand_landmarks)

            fingers_text = f"Fingers: {fingers} | Hand: {hand_label}"

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
        fingers_text,
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (30, 145),
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

cap.release()
cv2.destroyAllWindows()
'@
Set-Content -Path $path -Value $content -Encoding UTF8
