# import cv2
# import numpy as np
# import math
# import os
# from datetime import datetime
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier

# class SignLanguageRecognizer:
#     def __init__(self, model_path: str, label_path: str, save_dir: str = "Data", img_size: int = 300, offset: int = 20, confidence_threshold: float = 70.0):
#         """
#         Initialize the SignLanguageRecognizer for gesture detection and recognition.
        
#         Args:
#             model_path (str): Path to the trained model file (.h5)
#             label_path (str): Path to the labels file
#             save_dir (str): Directory to save collected images
#             img_size (int): Size of the output square image
#             offset (int): Padding around the hand ROI
#             confidence_threshold (float): Minimum confidence for valid predictions
#         """
#         self.img_size = img_size
#         self.offset = offset
#         self.save_dir = save_dir
#         self.counter = 0
#         self.confidence_threshold = confidence_threshold
        
#         # Load the trained classifier
#         self.classifier = Classifier(model_path, label_path)
        
#         # Load labels
#         with open(label_path, "r") as f:
#             self.labels = [line.strip() for line in f.readlines()]
        
#         # Initialize video capture
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             raise RuntimeError("Failed to open video capture device")
        
#         # Initialize hand detector
#         self.detector = HandDetector(maxHands=2)  # Detect up to 2 hands
        
#         # Ensure save directories exist
#         os.makedirs(save_dir, exist_ok=True)
#         os.makedirs(os.path.join(save_dir, "Collected"), exist_ok=True)
#         os.makedirs(os.path.join(save_dir, "Predicted"), exist_ok=True)
    
#     def process_hand(self, img: np.ndarray, hand: dict) -> np.ndarray:
#         """
#         Process detected hand and create standardized image for classification.
        
#         Args:
#             img: Input image
#             hand: Hand detection data
            
#         Returns:
#             Preprocessed image ready for classification
#         """
#         x, y, w, h = hand['bbox']
        
#         # Create white background image
#         img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
        
#         # Crop hand region
#         img_crop = img[max(y - self.offset, 0):min(y + h + self.offset, img.shape[0]),
#                       max(x - self.offset, 0):min(x + w + self.offset, img.shape[1])]
        
#         if img_crop.size == 0:
#             raise ValueError("Invalid crop region")
        
#         # Calculate aspect ratio and resize
#         aspect_ratio = h / w
        
#         if aspect_ratio > 1:
#             k = self.img_size / h
#             w_cal = math.ceil(k * w)
#             img_resize = cv2.resize(img_crop, (w_cal, self.img_size))
#             w_gap = math.ceil((self.img_size - w_cal) / 2)
#             img_white[:, w_gap:w_cal + w_gap] = img_resize
#         else:
#             k = self.img_size / w
#             h_cal = math.ceil(k * h)
#             img_resize = cv2.resize(img_crop, (self.img_size, h_cal))
#             h_gap = math.ceil((self.img_size - h_cal) / 2)
#             img_white[h_gap:h_cal + h_gap, :] = img_resize
            
#         return img_white
    
    
    
#     def is_fingers_closed(self, hand):
#         """
#         Checks if only thumb, index, and middle fingers are open, and other fingers are closed (used for 'no' gesture).
        
#         Args:
#             hand: Hand detection data
        
#         Returns:
#             bool: True if thumb, index, and middle fingers are open and other fingers are closed
#         """
#         landmarks = hand['lmList']
#         thumb_open = self.calculate_distance(landmarks[4], landmarks[0]) > 50
#         index_open = self.calculate_distance(landmarks[8], landmarks[0]) > 50
#         middle_open = self.calculate_distance(landmarks[12], landmarks[0]) > 50
#         ring_closed = self.calculate_distance(landmarks[16], landmarks[0]) < 30
#         pinky_closed = self.calculate_distance(landmarks[20], landmarks[0]) < 30
        
#         return thumb_open and index_open and middle_open and ring_closed and pinky_closed
    
#     def is_please(self, hand):
#         """
#         Checks if 'please' gesture (four fingers open and one closed) is made.
        
#         Args:
#             hand: Hand detection data
        
#         Returns:
#             bool: True if the gesture is 'please'
#         """
#         landmarks = hand['lmList']
        
#         # Check that only the thumb is closed and the other fingers are open
#         thumb_closed = self.calculate_distance(landmarks[4], landmarks[0]) < 30
#         index_open = self.calculate_distance(landmarks[8], landmarks[0]) > 50
#         middle_open = self.calculate_distance(landmarks[12], landmarks[0]) > 50
#         ring_open = self.calculate_distance(landmarks[16], landmarks[0]) > 50
#         pinky_open = self.calculate_distance(landmarks[20], landmarks[0]) > 50
        
#         return thumb_closed and index_open and middle_open and ring_open and pinky_open
    
#     def is_okay(self, hand):
#         """
#         Checks if 'okay' gesture (thumb and index forming O, others open).
        
#         Args:
#             hand: Hand detection data
        
#         Returns:
#             bool: True if the gesture is 'okay'
#         """
#         landmarks = hand['lmList']
        
#         # Check if thumb and index form a circle, and other fingers are open
#         thumb_index_touching = self.calculate_distance(landmarks[4], landmarks[8]) < 30
#         other_fingers_open = all(self.calculate_distance(landmarks[i], landmarks[0]) > 50 for i in [12, 16, 20])
        
#         return thumb_index_touching and other_fingers_open
    
#     def is_iloveyou(self, hand):
#         """
#         Checks if 'I love you' gesture (thumb, index, and pinky open, others closed).
        
#         Args:
#             hand: Hand detection data
        
#         Returns:
#             bool: True if the gesture is 'I love you'
#         """
#         landmarks = hand['lmList']
        
#         # Check thumb, index, and pinky open, middle and ring closed
#         thumb_open = self.calculate_distance(landmarks[4], landmarks[0]) > 50
#         index_open = self.calculate_distance(landmarks[8], landmarks[0]) > 50
#         pinky_open = self.calculate_distance(landmarks[20], landmarks[0]) > 50
#         middle_closed = self.calculate_distance(landmarks[12], landmarks[0]) < 30
#         ring_closed = self.calculate_distance(landmarks[16], landmarks[0]) < 30
        
#         return thumb_open and index_open and pinky_open and middle_closed and ring_closed
    
#     def calculate_distance(self, point1, point2):
#         """
#         Calculate the Euclidean distance between two points.
        
#         Args:
#             point1: First point
#             point2: Second point
        
#         Returns:
#             float: Distance between the points
#         """
#         return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    
#     def recognize_sign(self):
#         """
#         Recognize sign language gestures in real-time and optionally save them.
        
#         Args:
#             folder_name: Folder name to save collected images (if needed)
#         """
#         os.makedirs(os.path.join(self.save_dir, "Predicted"), exist_ok=True)
        
#         try:
#             while True:
#                 success, img = self.cap.read()
#                 if not success:
#                     print("Failed to read frame")
#                     break
                
#                 hands, img = self.detector.findHands(img)
#                 img_output = img.copy()
                
#                 if hands:
#                     for hand in hands:
#                         try:
#                             img_white = self.process_hand(img, hand)
#                             prediction, index = self.classifier.getPrediction(img_white, draw=False)
                            

#                             # Get label and confidence
#                             label = self.labels[index]
#                             confidence = round(prediction[index] * 100, 2)
                            
#                             # Display the prediction on the image
#                             x, y, w, h = hand['bbox']
#                             cv2.rectangle(img_output, (x - self.offset, y - self.offset - 50),
#                                           (x + w + self.offset, y - self.offset), (0, 255, 0), cv2.FILLED)
#                             cv2.putText(img_output, f"{label} ({confidence}%)", (x, y - 30),
#                                         cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
#                             cv2.rectangle(img_output, (x - self.offset, y - self.offset),
#                                           (x + w + self.offset, y + h + self.offset), (0, 255, 0), 4)
                            
#                             # Save image for predicted sign
#                             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                             sign_folder = os.path.join(self.save_dir, "Predicted", label)
#                             os.makedirs(sign_folder, exist_ok=True)
#                             save_path = os.path.join(sign_folder, f"{label}_{timestamp}.jpg")
#                             cv2.imwrite(save_path, img_white)
#                             print(f"Saved image for {label} at {save_path}")
                            
#                             # Save image when 's' is pressed in the Collected folder
#                             key = cv2.waitKey(1)
#                             if key == ord('s'):
#                                 collected_folder = os.path.join(self.save_dir, "Collected", label)
#                                 os.makedirs(collected_folder, exist_ok=True)
#                                 save_path = os.path.join(collected_folder, f"Image_{self.counter}_{timestamp}.jpg")
#                                 cv2.imwrite(save_path, img_white)
#                                 self.counter += 1
#                                 print(f"Saved collected image {self.counter}")
#                             elif key == 27:  # ESC to exit
#                                 return
#                         except Exception as e:
#                             print(f"Error processing hand: {e}")
#                             continue
                
#                 cv2.imshow("Sign Language Recognition", img_output)
#                 if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
#                     break
#         finally:
#             self.cleanup()
    
#     def cleanup(self):sa
#         """Release resources and close windows."""
#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Replace with your model and label paths
#     MODEL_PATH = "Model/keras_model.h5"
#     LABEL_PATH = "Model/labels.txt"
    
#     recognizer = SignLanguageRecognizer(MODEL_PATH, LABEL_PATH)
#     recognizer.recognize_sign()

import cv2
import numpy as np
import math
import os



from datetime import datetime
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier



class SignLanguageRecognizer:
    def __init__(self, model_path: str, label_path: str, save_dir: str = "Data", img_size: int = 300, offset: int = 20, confidence_threshold: float = 70.0):
        """
        Initialize the SignLanguageRecognizer for gesture detection and recognition.
        """
        self.img_size = img_size
        self.offset = offset
        self.save_dir = save_dir
        self.counter = 0
        self.confidence_threshold = confidence_threshold

        # Load the trained classifier
        self.classifier = Classifier(model_path, label_path)

        # Load labels
        with open(label_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video capture device")

        # Initialize hand detector
        self.detector = HandDetector(maxHands=2)

        # Ensure save directories exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "Collected"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "Predicted"), exist_ok=True)

    def process_hand(self, img: np.ndarray, hand: dict) -> np.ndarray:
        """
        Process detected hand and create standardized image for classification.
        """
        x, y, w, h = hand['bbox']
        img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
        img_crop = img[max(y - self.offset, 0):min(y + h + self.offset, img.shape[0]),
                      max(x - self.offset, 0):min(x + w + self.offset, img.shape[1])]

        if img_crop.size == 0:
            raise ValueError("Invalid crop region")

        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = self.img_size / h
            w_cal = math.ceil(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, self.img_size))
            w_gap = math.ceil((self.img_size - w_cal) / 2)
            img_white[:, w_gap:w_cal + w_gap] = img_resize
        else:
            k = self.img_size / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (self.img_size, h_cal))
            h_gap = math.ceil((self.img_size - h_cal) / 2)
            img_white[h_gap:h_cal + h_gap, :] = img_resize

        return img_white

    def calculate_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def is_hello(self, hand):
        """
        Checks if 'hello' gesture (open palm) is made.
        """
        landmarks = hand['lmList']
        all_open = all(self.calculate_distance(landmarks[i], landmarks[0]) > 50 for i in [4, 8, 12, 16, 20])
        return all_open

    def is_fingers_closed(self, hand):
        """
        Checks if thumb, index, and middle fingers are open, and others are closed.
        """
        landmarks = hand['lmList']
        thumb_open = self.calculate_distance(landmarks[4], landmarks[0]) > 50
        index_open = self.calculate_distance(landmarks[8], landmarks[0]) > 50
        middle_open = self.calculate_distance(landmarks[12], landmarks[0]) > 50
        ring_closed = self.calculate_distance(landmarks[16], landmarks[0]) < 30
        pinky_closed = self.calculate_distance(landmarks[20], landmarks[0]) < 30
        return thumb_open and index_open and middle_open and ring_closed and pinky_closed

    def is_please(self, hand):
        landmarks = hand['lmList']
        return (
            self.calculate_distance(landmarks[4], landmarks[0]) < 30 and
            all(self.calculate_distance(landmarks[i], landmarks[0]) > 50 for i in [8, 12, 16, 20])
        )

    def is_okay(self, hand):
        """
        Checks if 'okay' gesture (thumb and index forming O, others open) is made.
        """
        landmarks = hand['lmList']
        thumb_index_touching = self.calculate_distance(landmarks[4], landmarks[8]) < 30
        others_open = all(self.calculate_distance(landmarks[i], landmarks[0]) > 50 for i in [12, 16, 20])
        return thumb_index_touching and others_open

    def is_iloveyou(self, hand):
        """
        Checks if 'I love you' gesture (thumb, index, and pinky open) is made.
        """
        landmarks = hand['lmList']
        thumb_open = self.calculate_distance(landmarks[4], landmarks[0]) > 50
        index_open = self.calculate_distance(landmarks[8], landmarks[0]) > 50
        pinky_open = self.calculate_distance(landmarks[20], landmarks[0]) > 50
        middle_closed = self.calculate_distance(landmarks[12], landmarks[0]) < 30
        ring_closed = self.calculate_distance(landmarks[16], landmarks[0]) < 30
        return thumb_open and index_open and pinky_open and middle_closed and ring_closed

    def recognize_sign(self):
        """
        Recognize sign language gestures in real-time and save them if required.
        """
        os.makedirs(os.path.join(self.save_dir, "Predicted"), exist_ok=True)

        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("Failed to read frame")
                    break

                hands, img = self.detector.findHands(img)
                img_output = img.copy()

                if hands:
                    for hand in hands:
                        try:
                            img_white = self.process_hand(img, hand)
                            prediction, index = self.classifier.getPrediction(img_white, draw=False)

                            # Custom gesture recognition
                            if self.is_hello(hand):
                                label, confidence = "Hello", 100
                            elif self.is_fingers_closed(hand):
                                label, confidence = "No", 100
                            elif self.is_please(hand):
                                label, confidence = "Please", 100
                            elif self.is_okay(hand):
                                label, confidence = "Okay", 100
                            elif self.is_iloveyou(hand):
                                label, confidence = "I Love You", 100
                            else:
                                label = self.labels[index]
                                confidence = round(prediction[index] * 100, 2)

                            # Display prediction
                            x, y, w, h = hand['bbox']
                            cv2.rectangle(img_output, (x - self.offset, y - self.offset - 50),
                                          (x + w + self.offset, y - self.offset), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img_output, f"{label} ({confidence}%)", (x, y - 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

                            # Save image
                            sign_folder = os.path.join(self.save_dir, "Predicted", label)
                            os.makedirs(sign_folder, exist_ok=True)
                            save_path = os.path.join(sign_folder, f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                            cv2.imwrite(save_path, img_white)
# Ensure image is saved
                            try:
                                print(f"Saving image to: {save_path}")
                                cv2.imwrite(save_path, img_white)
                            except Exception as e:
                                print(f"Error saving image: {e}")

                        except Exception as e:
                            print(f"Error processing hand: {e}")
                            continue

                cv2.imshow("Sign Language Recognition", img_output)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources and close windows."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "Model/keras_model.h5"
    LABEL_PATH = "Model/labels.txt"
    recognizer = SignLanguageRecognizer(MODEL_PATH, LABEL_PATH)
    recognizer.recognize_sign()
