import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import cv2

st.title("Handwritten Digit Classifier")

# Load or train a model (for example purposes, load pretrained MNIST model)
@st.cache_resource
def load_mnist_model():
    model = tf.keras.models.load_model('mnist_cnn.h5')
    return model

model = load_mnist_model()

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape (1, 28, 28, 1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.image(img.reshape(28, 28), caption="Processed Image", width=150)
    st.write(f"Predicted digit: {predicted_class}")

