import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import time
import math
from collections import Counter

# Streamlit Page Configuration
st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

# Paths and configurations
data_path = "C:/Users/ADMIN/Downloads/hand_sign(2)/hand_sign(1)/hand_sign/Data"
model_path = 'hand_sign_model.h5'
img_size = (300, 300)
offset = 20
detector = HandDetector(maxHands=1)

# Home Page
def home():
    st.title("🤟 Sign Language Recognition Project 🤟")

    # Display the image
    st.image("asl1.jpg", caption="Sign Language Recognition", use_column_width=False,width=1000)

# Data Collection Functionality
def data_collection():
    st.title("Data Collection")
    st.write("Press 'Start Webcam' to begin data collection using your webcam.")
    
    # Button to start and stop the webcam
    start_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop Webcam")

    if start_webcam and not st.session_state.get('webcam_running', False):
        cap = cv2.VideoCapture(0)
        st.session_state['webcam_running'] = True  # Set the webcam running state
        labels = [chr(i) for i in range(65, 91)]  # A-Z
        folder = data_path

        # Create folders if they don't exist
        for label in labels:
            if not os.path.exists(f"{folder}/{label}"):
                os.makedirs(f"{folder}/{label}")

        counter = 0
        st.write("Press 's' to save images.")
        
        while st.session_state.get('webcam_running', False):
            success, img = cap.read()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((300, 300, 3), np.uint8) * 255
                imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

                aspectRatio = h / w
                if aspectRatio > 1:
                    k = 300 / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, 300))
                    wGap = math.ceil((300 - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = 300 / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (300, hCal))
                    hGap = math.ceil((300 - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == ord('s'):
                label = st.text_input("Enter the label (A-Z): ").upper()
                if label in labels:
                    counter += 1
                    cv2.imwrite(f"{folder}/{label}/Image_{time.time()}.jpg", imgWhite)
                    st.write(f"Saved image {counter} for {label}")
                else:
                    st.write("Invalid label. Please enter a valid letter (A-Z).")

            if stop_webcam:  # Stop webcam if button pressed
                st.session_state['webcam_running'] = False
                break

        cap.release()
        cv2.destroyAllWindows()
        st.session_state['webcam_running'] = False  # Reset the webcam state


# EDA Functionality
def eda():
    st.title("Exploratory Data Analysis (EDA)")
    class_counts = Counter()
    image_sizes = []
    intensity_distributions = []
    missing_images = []

    for class_folder in os.listdir(data_path):
        class_folder_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_folder_path):
            class_counts[class_folder] += len(os.listdir(class_folder_path))
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        missing_images.append(image_path)
                        continue
                    image_sizes.append(img.shape)
                    intensity_distributions.append(img.flatten())
                except Exception as e:
                    missing_images.append(image_path)

    st.write("### Class Distribution")
    st.bar_chart(class_counts)

    st.write("### Image Sizes and Resolutions")
    unique_sizes = Counter(image_sizes)
    st.write(unique_sizes)

    st.write("### Image Intensity Distribution")
    intensity_distributions = np.concatenate(intensity_distributions, axis=0)
    fig, ax = plt.subplots()
    ax.hist(intensity_distributions, bins=50, color='blue', alpha=0.7)
    st.pyplot(fig)

    if missing_images:
        st.write(f"Missing or corrupt images: {len(missing_images)}")
    else:
        st.write("No missing or corrupt images found.")

# Training Functionality
def train_model():
    st.title("Model Training")
    
    if st.button("Start Training"):
        import tensorflow as tf
        from tensorflow import keras
        from keras.preprocessing.image import ImageDataGenerator
        from keras.models import Sequential
        from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from keras.models import load_model

        # Path to data folder
        data_path = "C:/Users/bhava/Desktop/ML PROJECT/hand_sign(1)/hand_sign/Data"
        
        # Image size and batch size
        img_size = (300, 300)
        batch_size = 32
        
        # Data generator with training and validation split
        datagen = ImageDataGenerator(
            rescale=1./255,        # Normalize the pixel values
            validation_split=0.2    # 20% validation data
        )

        # Load training data
        train_data = datagen.flow_from_directory(
            data_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        # Load validation data
        val_data = datagen.flow_from_directory(
            data_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        # Build the CNN model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),  # Adding Dropout layer for regularization
            Dense(26, activation='softmax')  # 26 classes for A-Z
        ])

        # Compile Model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        st.write("Training Started...")

        # Train Model
        model.fit(train_data, validation_data=val_data, epochs=10)
        model.save(model_path)
        st.write("Training Completed. Model Saved as `hand_sign_model.h5`.")

# Testing Functionality
def test_model():
    st.title("Test the Model")
    st.write("Press 'Start Webcam' to begin testing.")
    
    # Button to start and stop the webcam
    start_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop Webcam")
    
    # Load the trained model and set up labels
    model = load_model('hand_sign_model.h5')
    labels = [chr(i) for i in range(65, 91)]  # A-Z
    detector = HandDetector(maxHands=1)
    imgSize = 300
    offset = 20

    if start_webcam and not st.session_state.get('webcam_running', False):
        cap = cv2.VideoCapture(0)
        st.session_state['webcam_running'] = True  # Set the webcam running state
        st.write("Press 'q' to stop testing.")

        # Webcam loop
        while st.session_state.get('webcam_running', False):
            success, img = cap.read()
            if not success:
                st.write("Failed to access webcam. Please check your webcam settings.")
                break
            
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
                cv2.putText(imgOutput, predicted_label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Output", imgOutput)
            key = cv2.waitKey(1)
            if stop_webcam:  # Stop webcam if button pressed
                st.session_state['webcam_running'] = False
                break
            
            if key == ord('q'):  # Exit if 'q' is pressed
                st.session_state['webcam_running'] = False
                break

        cap.release()
        cv2.destroyAllWindows()
        st.session_state['webcam_running'] = False  # Reset the webcam state

# Main Application Logic
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ("Home", "Data Collection", "EDA", "Train Model", "Test Model"))

if page == "Home":
    home()
elif page == "Data Collection":
    data_collection()
elif page == "EDA":
    eda()
elif page == "Train Model":
    train_model()
elif page == "Test Model":
    test_model()

# Sidebar Instructions
with st.sidebar:
    st.write("---")
    st.title("🔍 How to Use")
    st.write("Navigate between the tasks:")
    st.write("1. **Data Collection**: Capture images using the webcam.")
    st.write("2. **EDA**: Explore the dataset statistics and graphs.")
    st.write("3. **Train Model**: Train the CNN model.")
    st.write("4. **Test Model**: Make predictions with the trained model.")
    st.write("---")
