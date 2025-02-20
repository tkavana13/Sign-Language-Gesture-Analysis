import os
import cv2
from datetime import datetime


class SignLanguageRecognizer:
    def __init__(self, detector, classifier, save_dir, labels=None, offset=20):
        self.detector = detector  # Hand detector (e.g., MediaPipe Hands)
        self.classifier = classifier  # Sign classifier model
        self.save_dir = save_dir
        self.offset = offset

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Error: Camera could not be opened.")

        # Labels for predictions (from the classifier)
        self.labels = labels if labels else ["Unknown"]

        # Create save directory
        os.makedirs(os.path.join(self.save_dir, "Predicted"), exist_ok=True)

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def is_hello(self, hand):
        """Check if 'hello' gesture (all fingers open) is made."""
        landmarks = hand['lmList']
        return all(self.calculate_distance(landmarks[i], landmarks[0]) > 50 for i in [4, 8, 12, 16, 20])

    def is_fingers_closed(self, hand):
        """Check if 'no' gesture (thumb, index, and middle open, others closed) is made."""
        landmarks = hand['lmList']
        return (
            self.calculate_distance(landmarks[4], landmarks[0]) > 50 and
            self.calculate_distance(landmarks[8], landmarks[0]) > 50 and
            self.calculate_distance(landmarks[12], landmarks[0]) > 50 and
            self.calculate_distance(landmarks[16], landmarks[0]) < 30 and
            self.calculate_distance(landmarks[20], landmarks[0]) < 30
        )

    def is_please(self, hand):
        """Check if 'please' gesture (four fingers open, thumb closed) is made."""
        landmarks = hand['lmList']
        thumb_closed = self.calculate_distance(landmarks[4], landmarks[0]) < 30
        all_open_except_thumb = all(self.calculate_distance(landmarks[i], landmarks[0]) > 50 for i in [8, 12, 16, 20])
        return thumb_closed and all_open_except_thumb

    def recognize_sign(self):
        """Recognize sign language gestures in real-time and save them."""
        try:
            while True:
                # Capture frame-by-frame
                success, img = self.cap.read()
                if not success:
                    print("Error: Failed to read frame.")
                    break

                # Detect hands
                hands, img = self.detector.findHands(img)
                img_output = img.copy()

                if hands:
                    for hand in hands:
                        try:
                            # Process the hand image
                            img_white = self.process_hand(img, hand)
                            if img_white is None:
                                continue  # Skip if no valid hand image

                            # Get prediction from classifier
                            prediction, index = self.classifier.getPrediction(img_white, draw=False)

                            # Custom gesture recognition
                            if self.is_hello(hand):
                                label, confidence = "Hello", 100
                            elif self.is_fingers_closed(hand):
                                label, confidence = "No", 100
                            elif self.is_please(hand):
                                label, confidence = "Please", 100
                            else:
                                label = self.labels[index] if index < len(self.labels) else "Unknown"
                                confidence = round(prediction[index] * 100, 2)

                            # Display prediction
                            x, y, w, h = hand['bbox']
                            self.display_prediction(img_output, label, confidence, x, y, w, h)

                            # Save image
                            self.save_image(img_white, label)

                        except Exception as e:
                            print(f"Error processing hand: {e}")

                # Display the result
                cv2.imshow("Sign Language Recognition", img_output)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

        finally:
            self.cleanup()

    def display_prediction(self, img, label, confidence, x, y, w, h):
        """Display the prediction label and confidence on the image."""
        cv2.rectangle(img, (x - self.offset, y - self.offset - 50),
                      (x + w + self.offset, y - self.offset), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{label} ({confidence}%)", (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    def save_image(self, img_white, label):
        """Save the processed hand image to the appropriate folder."""
        sign_folder = os.path.join(self.save_dir, "Predicted", label)
        os.makedirs(sign_folder, exist_ok=True)
        save_path = os.path.join(sign_folder, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(save_path, img_white)
        print(f"Image saved to: {save_path}")

    def cleanup(self):
        """Release camera and close windows."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def process_hand(self, img, hand):
        """Process and return the hand image for classification."""
        # Example of processing: crop and resize hand area
        x, y, w, h = hand['bbox']
        img_crop = img[max(y - self.offset, 0): y + h + self.offset, max(x - self.offset, 0): x + w + self.offset]
        img_resized = cv2.resize(img_crop, (224, 224))  # Example size, match classifier input
        return img_resized


# Usage Example
# Initialize your detector and classifier here
detector = None  # Replace with your hand detector
classifier = None  # Replace with your sign classifier
save_dir = "path_to_save_directory"
labels = ["Hello", "No", "Please", "Okay", "I Love You"]  # Example labels

recognizer = SignLanguageRecognizer(detector, classifier, save_dir, labels)
recognizer.recognize_sign()
