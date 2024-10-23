import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import math
import time
            
# Setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Set your output folder path
folder = "C:/Users/bhava/Desktop/ML PROJECT/hand_sign(1)/hand_sign/Data/A"  # Change this to the folder path where you want to save data
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
offset = 20
imgSize = 300
counter = 0

# Create folders if they don't exist
for label in labels:
    if not os.path.exists(f"{folder}/{label}"):
        os.makedirs(f"{folder}/{label}")

# Start capturing hand signs
print("Press 's' to save images, 'q' to quit")
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    
    # Save the images when 's' is pressed
    key = cv2.waitKey(1)
    if key == ord('s'):
        label = input("Enter the label: ").upper()
        if label in labels:
            counter += 1
            cv2.imwrite(f"{folder}/{label}/Image_{time.time()}.jpg", imgWhite)
            print(f"Saved image {counter} for {label}")
        else:
            print(f"Invalid label. Please enter a valid letter (A-Z).")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
