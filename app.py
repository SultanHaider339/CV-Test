import streamlit as st
import cv2
from PIL import Image
import numpy as np
from transformers import pipeline
import torch

# Page config
st.set_page_config(page_title="Real-Time Object Detection", layout="wide")

# Cache the model to avoid reloading
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("object-detection", model="facebook/detr-resnet-50", device=device)

# Load model
detector = load_model()

# Title
st.title("🎥 Real-Time Object Detection")
st.markdown("Using **DETR (Detection Transformer)** from Hugging Face")

# Sidebar controls
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
run_detection = st.sidebar.checkbox("Start Detection", value=False)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.subheader("Detection Stats")
    stats_placeholder = st.empty()

# Webcam feed
if run_detection:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot access webcam. Please check your camera permissions.")
    else:
        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Perform detection
            detections = detector(pil_image)
            
            # Filter by confidence
            filtered_detections = [d for d in detections if d['score'] >= confidence_threshold]
            
            # Draw bounding boxes
            frame_copy = frame_rgb.copy()
            detection_count = {}
            
            for detection in filtered_detections:
                box = detection['box']
                label = detection['label']
                score = detection['score']
                
                # Count objects
                detection_count[label] = detection_count.get(label, 0) + 1
                
                # Extract coordinates
                xmin = int(box['xmin'])
                ymin = int(box['ymin'])
                xmax = int(box['xmax'])
                ymax = int(box['ymax'])
                
                # Draw rectangle
                cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Put text
                text = f"{label}: {score:.2f}"
                cv2.putText(frame_copy, text, (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display frame
            frame_placeholder.image(frame_copy, channels="RGB", use_container_width=True)
            
            # Display stats
            with stats_placeholder.container():
                st.metric("Total Objects", len(filtered_detections))
                if detection_count:
                    st.write("**Detected Objects:**")
                    for obj, count in detection_count.items():
                        st.write(f"• {obj}: {count}")
                else:
                    st.info("No objects detected")
        
        cap.release()
else:
    st.info("👈 Check 'Start Detection' in the sidebar to begin")
    st.image("https://via.placeholder.com/800x600.png?text=Webcam+Feed+Will+Appear+Here", 
             use_container_width=True)
