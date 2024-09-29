import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your YOLOv8m model (ensure the path to your model is correct)
model = YOLO('YOLOV8m.pt')  # Replace with your actual model file name

# Streamlit app title
st.title("ECG Signal Classification")

# Upload an image (assumes your ECG signals are visualized as images)
uploaded_file = st.file_uploader("Upload an ECG signal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG Signal", use_column_width=True)

    # Perform inference
    results = model(image)

    # Display results
    st.write("Classifications: ", results.names)
    st.write("Confidences: ", [f"{score:.2f}" for score in results.pred[0][:, 4].cpu().numpy()])

    # Display the processed image with bounding boxes (if relevant)
    results.show()  # This will display the bounding boxes on the image if any
else:
    st.write("Please upload an ECG image for classification.")
