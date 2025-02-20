import cv2
from cvzone.HandTrackingModule import HandDetector

class GestureRecognizer:
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.cap = cv2.VideoCapture(0)

    def recognize_sign(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to read from the camera.")
                break

            # Detect hands
            hands, img = self.detector.findHands(img)  # Returns hands and annotated image

            # Display the video feed
            cv2.imshow("Hand Gesture Recognition", img)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.recognize_sign()


