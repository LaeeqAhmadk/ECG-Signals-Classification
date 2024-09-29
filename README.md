# ECG Signal Classification Using YOLOv8m

Welcome to the ECG Signal Classification project, where we leverage the power of deep learning and object detection to automatically classify ECG signals as either "Normal" or "Abnormal." This project uses the YOLOv8m model trained on ECG images and is deployed on Hugging Face Spaces using Streamlit for an interactive and user-friendly experience.

# Project Overview

Electrocardiograms (ECGs) are crucial for diagnosing heart conditions. However, manual interpretation can be error-prone and time-consuming. This project automates the classification of ECG signals, detecting abnormalities with high accuracy.

We use a dataset from Roboflow, and the model is based on YOLOv8m, a state-of-the-art object detection architecture. The app is designed to classify ECG images as either:

- Normal (for healthy heart signals)
- Abnormal (for heart conditions like Myocardial Infarction or Heart Block)

# Goal:

To provide healthcare professionals with an easy-to-use, fast, and accurate tool for ECG signal classification, potentially reducing the diagnostic workload and improving patient care.

# Features

- Real-time ECG Classification: Upload an ECG image, and the model will classify it as "Normal" or "Abnormal" in seconds.
- Confidence Scores: Get a detailed confidence score for each prediction.
- Streamlit Web App: Easy-to-use interface deployed on Hugging Face Spaces.
- Visual Feedback: Bounding boxes on detected signals to give insights into the predictions.

# Dataset

-  Source: **Roboflow ECG Classification Dataset**
- Classes:
- ECG HB (Heart Block)
- History_MI (History of Myocardial Infarction)
- MI-ECG (Myocardial Infarction)
- Normal-ECG
- The model simplifies these classes into:
- Normal (for "Normal-ECG")
- Abnormal (for all other classes)

# Model
Architecture: YOLOv8m
Framework: Ultralytics YOLO
Training: The model was trained on ECG images using bounding boxes to detect areas of interest in ECG signals and classify them into one of the four categories.

# Live Demo

The project is deployed on Hugging Face Spaces using Streamlit. You can try it out live using the link below:
https://huggingface.co/spaces/Laeeeq/ECG_Signals_Classification

# Results
The model achieves accurate classification with high confidence for both normal and abnormal ECG signals. It provides visual feedback through bounding boxes, making it easier to interpret the results.

# Deployment

This project is deployed using Streamlit on Hugging Face Spaces, allowing real-time access to the model via a web-based interface.

# Acknowledgments

- Roboflow for providing the ECG dataset.
- Ultralytics for the YOLOv8 model framework.
- Hugging Face for hosting the deployment.

