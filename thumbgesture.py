import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_index_dx = thumb_tip.x - index_tip.x
            thumb_index_dy = thumb_tip.y - index_tip.y
            angle = math.atan2(thumb_index_dy, thumb_index_dx)
            angle = math.degrees(angle)
            angle = angle + 360 if angle < 0 else angle

            direction = None
            if 45 <= angle < 135:
                direction = " Move Down"
            elif 135 <= angle < 225:
                direction = "Move Right"
            elif 225 <= angle < 315:
                direction = "Move Up"
            elif angle >= 315 or angle < 45:
                direction = "Move Left"
            else:
                direction = "No Gesture Detected.."

            cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        direction = "No Gesture Detected.."
        cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
