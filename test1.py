import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize video capture, hand detector, and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow detection of up to 2 hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants
offset = 20
imgSize = 300

# Update with the extended set of labels
labels = [
    "Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"
]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to access the webcam.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        for hand in hands:  # Process each detected hand
            x, y, w, h = hand['bbox']

            # Ensure bounding box doesn't go out of bounds
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:  # Ensure imgCrop is valid
                continue

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            try:
                if aspectRatio > 1:  # Height > Width
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:  # Width >= Height
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Get predictions
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Display label and bounding box for the current hand
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x + w + offset, y - offset), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 4)

            except Exception as e:
                print(f"Error in processing: {e}")
                continue

    cv2.imshow('Image', imgOutput)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
