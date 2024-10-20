import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow import keras
from keras.models import load_model
import numpy as np
import math

# Load the trained model
model = load_model('hand_sign_model.h5')

# Labels corresponding to the classes (A-Z)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Setup video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 300
offset = 20

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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

        # Prediction
        imgWhite_expanded = np.expand_dims(imgWhite, axis=0)
        prediction = model.predict(imgWhite_expanded)
        predicted_label = labels[np.argmax(prediction)]

        # Display the predicted label
        cv2.putText(imgOutput, predicted_label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        
        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
