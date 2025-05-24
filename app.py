import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from PIL import Image
import cv2

@st.cache(allow_output_mutation=True)
def load_and_train_model(binary=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize images
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # For binary classification, map digits to 0 or 1 (example: digit '0' vs others)
    if binary:
        y_train = np.where(y_train == 0, 1, 0)
        y_test = np.where(y_test == 0, 1, 0)
        num_classes = 1
    else:
        num_classes = 10

    # Reshape input for CNN
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = Sequential([
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='sigmoid' if binary else 'softmax')
    ])

    loss = 'binary_crossentropy' if binary else 'sparse_categorical_crossentropy'

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

    return model

def preprocess_image(image):
    img = image.convert('L')  # grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    img = img / 255.0
    img = img[np.newaxis, ..., np.newaxis]
    return img

def main():
    st.title("Handwritten Digit Classification")

    mode = st.radio("Choose Classification Mode:", ["Multi-class (0-9 digits)", "Binary (digit '0' vs others)"])

    binary = (mode == "Binary (digit '0' vs others)")

    model = load_and_train_model(binary=binary)

    uploaded_file = st.file_uploader("Upload a handwritten digit image (PNG/JPG)")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=150)

        x = preprocess_image(img)
        preds = model.predict(x)

        if binary:
            pred_label = "Digit 0" if preds[0][0] > 0.5 else "Other Digit"
            st.write(f"Prediction: {pred_label} (confidence: {preds[0][0]:.2f})")
        else:
            pred_digit = np.argmax(preds)
            st.write(f"Predicted Digit: {pred_digit}")

if __name__ == "__main__":
    main()
