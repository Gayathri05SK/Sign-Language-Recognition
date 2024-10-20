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
    Dropout(0.5),
    Dense(26, activation='softmax')  # 26 output neurons for A-Z
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_data,
    epochs=10,  # Adjust based on the size of your dataset
    validation_data=val_data
)

# Save the model
model.save('hand_sign_model.h5')

print("Model trained and saved as hand_sign_model.h5")
