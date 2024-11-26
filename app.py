import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2

# Load the trained model
model = load_model("waste_classification_model.h5")

# Define the class labels
classes = {0: "Organic Waste (O)", 1: "Inorganic Waste (R)"}

# Add custom CSS to enhance UI design with smaller font sizes
st.markdown(
    """
    <style>
    body {
        background: rgba(255, 255, 255, 0.8);
        font-family: Arial, sans-serif;
        color: #fff; /* Text color to contrast with the background */
    }
    .main {
        background-image: linear-gradient(to right, #5a85c6, #248ec5, #0095ba, #009aa9, #1a9d94);
        border-radius: 15px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 15px;
        cursor: pointer;
        font-size: 12px;  /* Further reduced font size */
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .header {
        font-family: 'Arial Black', sans-serif;
        font-size: 28px;  /* Reduced font size */
        color: #4CAF50;
        text-align: center;
    }
    .subheader {
        font-family: 'Arial', sans-serif;
        font-size: 18px;  /* Further reduced font size */
        color: #ffffff;
        text-align: center;
    }
    hr {
        border: none;
        height: 1px;
        background: #ddd;
        margin: 15px 0;
    }
    .markdown-container {
        padding: 15px;
        background-color: #2e2e2e;
        color: #ffffff;
        border-radius: 8px;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        font-size: 14px;  /* Reduced font size for readme */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="header">🌿 Waste Classification App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Classify waste as Organic or Inorganic</div>', unsafe_allow_html=True)

# Sidebar for mode selection
st.sidebar.title("⚙️ Input Mode")
input_mode = st.sidebar.radio("Choose input mode:", ["Upload Image", "Camera Capture"])

# Preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to the model input size
    image_array = img_to_array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Predict function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return classes[predicted_class], confidence

# Draw bounding box and label on the image
def draw_bounding_box(image, label, confidence):
    image_np = np.array(image)  # Convert PIL image to numpy array
    h, w, _ = image_np.shape

    # Draw bounding box
    x1, y1, x2, y2 = int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box

    # Add label
    label_text = f"{label} ({confidence:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label_text, font, 0.7, 2)[0]
    text_x = x1
    text_y = y1 - 10
    cv2.rectangle(image_np, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
    cv2.putText(image_np, label_text, (text_x, text_y), font, 0.7, (0, 0, 0), 2)

    return Image.fromarray(image_np)  # Convert back to PIL Image

# Main functionality
if input_mode == "Upload Image":
    st.write("📤 **Upload Image**")
    uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict and show result
        if st.button("🔍 Classify Image"):
            label, confidence = predict(image)
            result_image = draw_bounding_box(image, label, confidence)
            st.image(result_image, caption="Result with Bounding Box", use_column_width=True)

elif input_mode == "Camera Capture":
    st.write("📸 **Capture Image**")
    picture = st.camera_input("Take a picture")
    if picture:
        # Display captured image
        image = Image.open(picture)
        st.image(image, caption="Captured Image", use_column_width=True)

        # Predict and show result
        if st.button("🔍 Classify Image"):
            label, confidence = predict(image)
            result_image = draw_bounding_box(image, label, confidence)
            st.image(result_image, caption="Result with Bounding Box", use_column_width=True)

# Footer
# Footer with personal links and acknowledgment
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-family: Arial, sans-serif; font-size: 12px;">  <!-- Further reduced font size -->
        🌟 Powered by <b style="font-size: 14px;">TensorFlow</b>, <b style="font-size: 14px;">OpenCV</b>, and <b style="font-size: 14px;">Streamlit</b><br><br>
        <b style="font-size: 14px;">Made by:</b> Hitesh Kumar<br>
        <a href="https://www.linkedin.com/in/hitesh-kumar-aiml/" target="_blank" style="color: #0a66c2; font-size: 14px;">LinkedIn</a> | 
        <a href="https://github.com/Hiteshydv001" target="_blank" style="color: #333; font-size: 14px;">GitHub</a><br><br>
        🎉<b style="font-size: 14px;">Edunet-Shell Skills4Future AICTE Internship Project</b> (Nov-Dec)<br>
        Focused on <b style="font-size: 14px;">Green Skills</b> & <b style="font-size: 14px;">AI</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

import requests

# Fetch the raw content of the README file
url = "https://raw.githubusercontent.com/Hiteshydv001/Waste-classification-model-cnn/main/README.md"
response = requests.get(url)

# Apply custom CSS for padding and styling
st.markdown(
    """
    <style>
    .markdown-container {
        padding: 12px;
        background-color: #2e2e2e;
        color: #ffffff;
        border-radius: 8px;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        font-size: 12px;  /* Reduced font size for README */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the README content with custom container
st.markdown(
    f'<div class="markdown-container">{response.text}</div>',
    unsafe_allow_html=True
)
