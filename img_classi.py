import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load pre-trained MobileNetV2 model and labels
@st.cache_resource()
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

def load_imagenet_labels():
    labels_path = tf.keras.utils.get_file(
        "ImageNetLabels.json",
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    )
    with open(labels_path) as f:
        labels = json.load(f)
    return {k: v[1] for k, v in labels.items()}

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image, model, labels):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    return [(class_id, class_name, labels.get(class_id, "Unknown"), score) for class_id, class_name, score in decoded_predictions]

# Streamlit UI
st.title("Image Classification App")
st.write("Upload an image, and the model will classify it.")

model = load_model()
imagenet_labels = load_imagenet_labels()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Classifying..."):
        predictions = predict(image, model, imagenet_labels)
    
    st.write("### Predictions:")
    for i, (class_id, class_name, real_name, score) in enumerate(predictions):
        st.write(f"{i+1}. {class_name} ({real_name}) - {score*100:.2f}%")
