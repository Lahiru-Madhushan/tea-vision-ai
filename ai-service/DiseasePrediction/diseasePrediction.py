import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Dense,
    Dropout, BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPool2D, Dense,
    Dropout, BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam

model = Sequential()

# -----------------------------
# Block 1
# -----------------------------
model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu',
    input_shape=(128,128,3)
))
model.add(BatchNormalization())

model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# -----------------------------
# Block 2
# -----------------------------
model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())

model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.30))

# -----------------------------
# Block 3
# -----------------------------
model.add(Conv2D(
    filters=128,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())

model.add(Conv2D(
    filters=128,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.35))

# -----------------------------
# Block 4
# -----------------------------
model.add(Conv2D(
    filters=256,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())

model.add(Conv2D(
    filters=256,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.40))

# -----------------------------
# Classifier
# -----------------------------
model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.load_weights('final_tea_model.h5')

def predict_disease(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the disease
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

   

    confidence = np.max(predictions)

    # Class names (update these to match your actual classes)
    class_names = [
        'Tea algal leaf spot',
        'Brown Blight', 
        'Gray Blight',
        'Helopeltis',
        'Red spider',
        'Green mirid bug',
        'Healthy leaf'
    ]

    print(f"Predicted Class Index: {predicted_class}")
    print(f"Predicted Disease: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2%} ({confidence*100:.2f}%)")

    class_name = class_names[predicted_class]



    return class_name , confidence

