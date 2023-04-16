import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def check_trained_model_exists(model_path):
    return os.path.isfile(model_path)


# Load the dataset using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Create the deep learning model
weights_path = "./mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)
base_model.load_weights(weights_path)


model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model_path = "weed_classification_karnataka_model.h5"

if not check_trained_model_exists(model_path):
    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
    # Save the trained model
    model.save(model_path)
else:
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)


# Mapping class indices to labels
labels = {v: k for k, v in train_generator.class_indices.items()}

# Set up a live webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (224, 224))
    input_frame = np.expand_dims(resized_frame, axis=0) / 255.0

    preds = model.predict(input_frame)
    predicted_class = labels[np.argmax(preds)]

    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Weed Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Use the test image for prediction
# test_image = cv2.imread('test.jpeg')
# resized_test_image = cv2.resize(test_image, (224, 224))
# input_test_image = np.expand_dims(resized_test_image, axis=0) / 255.0

# preds = model.predict(input_test_image)
# predicted_class = labels[np.argmax(preds)]

# cv2.putText(test_image, predicted_class, (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# cv2.imshow('Weed Classification', test_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
