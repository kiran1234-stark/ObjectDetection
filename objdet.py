import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import torch
from torchvision import models, transforms

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # Use the YOLOv8n model

# Load MobileNet SSD model
@st.cache_resource
def load_mobilenet_ssd():
    prototxt_path = r"D:/Kiki/pg/ObjectDet/Models/MobileNet SSD/deploy.prototxt"
    model_path = r"D:/Kiki/pg/ObjectDet/Models/MobileNet SSD/mobilenet_iter_73000.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return model

# Load Faster R-CNN model
@st.cache_resource
def load_faster_rcnn():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Function to perform object detection
def perform_detection(image, model_name, conf_threshold):
    start_time = time.time()
    
    if model_name == "YOLOv8":
        model = load_yolo_model()
        results = model(image, conf=conf_threshold)  # Run inference
        output_image = results[0].plot()
    
    elif model_name == "MobileNet SSD":
        model = load_mobilenet_ssd()
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
        model.setInput(blob)
        detections = model.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        output_image = image
    
    elif model_name == "Faster R-CNN":
        model = load_faster_rcnn()
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            predictions = model(image_tensor)
        
        for i in range(len(predictions[0]['boxes'])):
            score = predictions[0]['scores'][i].item()
            if score > conf_threshold:
                box = predictions[0]['boxes'][i].detach().numpy().astype(int)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        output_image = image
    
    else:
        st.error("Invalid model selection.")
        return None, None
    
    inference_time = time.time() - start_time
    return output_image, inference_time

# Streamlit app
st.title("Object Detection App")

# Sidebar for model selection and settings
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Select a model:", ["YOLOv8", "MobileNet SSD", "Faster R-CNN"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
st.sidebar.write("Selected Confidence:", conf_threshold)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to OpenCV format
    image_np = np.array(image)

    # Perform detection
    st.subheader("Detection Results")
    result_image, inference_time = perform_detection(image_np, model_name, conf_threshold)

    if result_image is not None:
        st.image(result_image, caption="Detected Objects", use_container_width=True)
        st.sidebar.header("Performance Metrics")
        st.sidebar.text(f"Inference Time: {inference_time:.3f} seconds")
