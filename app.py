import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load your YOLOv8m model (ensure the path to your model is correct)
model = YOLO('YOLOV8m.pt')  # Replace with your actual model file name

# Streamlit app title
st.title("ECG Signal Classification: Normal vs Abnormal")

# Upload an image (assumes your ECG signals are visualized as images)
uploaded_file = st.file_uploader("Upload an ECG signal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG Signal", use_column_width=True)

    # Perform inference
    results = model(image)  # Make sure the image is compatible with the model (resize if needed)

    # Assuming the model provides results with bounding boxes and classes
    if results and len(results) > 0:
        result = results[0]  # Get the first result from the list

        # Extract information from the result
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs (0: Normal, 1: Abnormal)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Assuming your model has 2 classes defined in the data.yaml file
        class_names = ["Normal", "Abnormal"]

        # Display results for each detected ECG classification
        for i, box in enumerate(boxes):
            predicted_class = int(class_ids[i])
            confidence_score = confidences[i]

            # Draw the bounding box and label
            st.write(f"Prediction: **{class_names[predicted_class]}**")
            st.write(f"Confidence Score: **{confidence_score:.2f}**")

            # Optionally draw bounding boxes on the image using Matplotlib
            fig, ax = plt.subplots()
            ax.imshow(image)
            # Draw bounding box (x1, y1, x2, y2)
            rect = plt.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(box[0], box[1] - 10, f"{class_names[predicted_class]}: {confidence_score:.2f}",
                    color='white', fontsize=12, backgroundcolor='r')
            st.pyplot(fig)
    else:
        st.write("No valid predictions found.")
else:
    st.write("Please upload an ECG image for classification.")
