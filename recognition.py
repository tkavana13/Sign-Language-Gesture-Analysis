import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition logic
def recognize_gesture(landmarks):
    """
    Recognize gesture based on hand landmarks.
    Returns a string indicating the recognized gesture.
    """
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y

    # Hello: All fingers are up
    if (
        index_tip < landmarks[6].y
        and middle_tip < landmarks[10].y
        and ring_tip < landmarks[14].y
        and pinky_tip < landmarks[18].y
    ):
        return "Hello"

    # Yes: Thumb up, other fingers down
    if (
        thumb_tip < landmarks[3].y
        and index_tip > landmarks[6].y
        and middle_tip > landmarks[10].y
        and ring_tip > landmarks[14].y
        and pinky_tip > landmarks[18].y
    ):
        return "Yes"

    # No: Index finger up, other fingers down
    if (
        index_tip < landmarks[6].y
        and middle_tip > landmarks[10].y
        and ring_tip > landmarks[14].y
        and pinky_tip > landmarks[18].y
    ):
        return "No"

    return "Unknown"

# Main function to run gesture detection
def main():
    cap = cv2.VideoCapture(0)  # Access webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Unable to access webcam. Exiting...")
                break

            # Flip and process the frame
            frame = cv2.flip(frame, 1)  # Mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for hand detection
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the screen
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Recognize gesture
                    gesture = recognize_gesture(hand_landmarks.landmark)

                    # Display the gesture on the screen
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # Display the frame
            cv2.imshow("Gesture Recognition", frame)

            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
