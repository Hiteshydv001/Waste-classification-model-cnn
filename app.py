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

# Function to switch between dark and light mode CSS
def get_mode_styles(is_dark_mode):
    if is_dark_mode:
        return """
            <style>
            body {
                background-color: #121212;
                color: #ffffff;
                font-family: Arial, sans-serif;
            }
            .main {
                background: linear-gradient(to right, #4B0082, #8A2BE2);
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .stButton button {
                background-color: #6200ea;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #3700b3;
            }
            .header {
                font-family: 'Arial Black', sans-serif;
                font-size: 35px;
                color: #bb86fc;
                text-align: center;
            }
            .subheader {
                font-family: 'Arial', sans-serif;
                font-size: 22px;
                color: #ffffff;
                text-align: center;
            }
            hr {
                border: none;
                height: 1px;
                background: #ddd;
                margin: 20px 0;
            }
            </style>
        """
    else:
        return """
            <style>
            body {
                background: rgba(255, 255, 255, 0.8);
                font-family: Arial, sans-serif;
                color: #000;
            }
            .main {
                background: linear-gradient(to right, #764BA2, #667EEA);
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
            .header {
                font-family: 'Arial Black', sans-serif;
                font-size: 35px;
                color: #4CAF50;
                text-align: center;
            }
            .subheader {
                font-family: 'Arial', sans-serif;
                font-size: 22px;
                color: #000;
                text-align: center;
            }
            hr {
                border: none;
                height: 1px;
                background: #ddd;
                margin: 20px 0;
            }
            </style>
        """

# Initial dark mode setting
is_dark_mode = st.sidebar.checkbox("Switch to Dark Mode", value=False)

# Apply CSS styles based on the mode
st.markdown(get_mode_styles(is_dark_mode), unsafe_allow_html=True)

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
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; font-family: Arial, sans-serif; font-size: 14px;">
        🌟 Powered by <b style="font-size: 16px;">TensorFlow</b>, <b style="font-size: 16px;">OpenCV</b>, and <b style="font-size: 16px;">Streamlit</b><br><br>
        <b style="font-size: 16px;">Made by:</b> Hitesh Kumar<br>
        <a href="https://www.linkedin.com/in/hitesh-kumar-aiml/" target="_blank" style="color: #333; font-size: 16px;">LinkedIn</a> | 
        <a href="https://github.com/Hiteshydv001" target="_blank" style="color: #333; font-size: 16px;">GitHub</a><br><br>
        🎉<b style="font-size: 16px;">Edunet-Shell Skills4Future AICTE Internship Project</b> (Nov-Dec)<br>
        Focused on <b style="font-size: 16px;">Green Skills</b> & <b style="font-size: 16px;">AI</b>.
    </div>
    """,
    unsafe_allow_html=True,
)
