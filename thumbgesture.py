import cv2
import mediapipe as mp
import math
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to get hand landmarks
    results = hands.process(frame_rgb)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks for the thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the angle between thumb-index finger line and horizontal line
            thumb_index_dx = thumb_tip.x - index_tip.x
            thumb_index_dy = thumb_tip.y - index_tip.y
            angle = math.atan2(thumb_index_dy, thumb_index_dx)
            angle = math.degrees(angle)
            angle = angle + 360 if angle < 0 else angle

            # Determine the direction based on angle
            direction = None
            if 45 <= angle < 135:
                direction = "Down"
            elif 135 <= angle < 225:
                direction = "Right"
            elif 225 <= angle < 315:
                direction = "Up"
            elif angle >= 315 or angle < 45:
                direction = "Left"

            # Draw direction on the frame
            cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
