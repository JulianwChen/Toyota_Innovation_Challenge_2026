import cv2
import mediapipe as mp
import time


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

def count_fingers_up(hand_landmarks, hand_label):
    """
    Returns a list showing which fingers are up.

    Output example:
    [1, 0, 0, 0, 0]

    Meaning:
    thumb up, index down, middle down, ring down, pinky down
    """

    fingers = []

    # Landmark IDs:
    # Thumb tip = 4
    # Index tip = 8
    # Middle tip = 12
    # Ring tip = 16
    # Pinky tip = 20

    tip_ids = [4, 8, 12, 16, 20]

    landmarks = hand_landmarks.landmark

    # -----------------------------
    # THUMB
    # -----------------------------
    # Thumb is different because it moves sideways.
    # For right hand, thumb tip is usually to the left of thumb joint when open.
    # For left hand, it is the opposite.

    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if landmarks[4].x > landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # -----------------------------
    # OTHER FOUR FINGERS
    # -----------------------------
    # For index/middle/ring/pinky:
    # If fingertip is higher on screen than middle joint, finger is up.
    # In image coordinates, smaller y = higher.

    for tip_id in tip_ids[1:]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


# -----------------------------
# BASIC GESTURE DECODER
# -----------------------------

def decode_gesture(fingers):
    """
    Convert finger states into simple gestures.
    """

    total_fingers = sum(fingers)

    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"

    if fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"

    if fingers == [0, 1, 0, 0, 0]:
        return "Pointing"

    if fingers == [0, 1, 1, 0, 0]:
        return "Peace Sign"

    if fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up / Thumb Out"

    return f"{total_fingers} fingers up"


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
            gesture_text = decode_gesture(fingers)

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
# -----------------------------

cap.release()
cv2.destroyAllWindows()