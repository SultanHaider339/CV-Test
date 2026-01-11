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
with st.spinner("Loading AI model..."):
    detector = load_model()

# Title
st.title("🎥 Real-Time Object Detection")
st.markdown("Using **DETR (Detection Transformer)** from Hugging Face")

# Sidebar controls
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Input method selection
input_method = st.sidebar.radio(
    "Input Source",
    ["📷 Upload Image", "🎥 Upload Video", "💻 Use Webcam (Local Only)"]
)

st.sidebar.info("💡 **Note:** Webcam only works when running locally. On Streamlit Cloud, use image/video upload.")

# Main content
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Detection Stats")
    stats_placeholder = st.empty()

# Function to process frame
def process_frame(frame_rgb, show_stats=True):
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
    
    # Display stats
    if show_stats:
        with stats_placeholder.container():
            st.metric("Total Objects", len(filtered_detections))
            if detection_count:
                st.write("**Detected Objects:**")
                for obj, count in detection_count.items():
                    st.write(f"• {obj}: {count}")
            else:
                st.info("No objects detected")
    
    return frame_copy

# Image Upload
if input_method == "📷 Upload Image":
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_rgb = np.array(image.convert('RGB'))
            
            # Process image
            with st.spinner("Detecting objects..."):
                result = process_frame(image_rgb)
            
            st.image(result, channels="RGB", use_container_width=True)
        else:
            st.info("👆 Upload an image to get started")

# Video Upload
elif input_method == "🎥 Upload Video":
    with col1:
        uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())
            
            # Process video
            cap = cv2.VideoCapture("temp_video.mp4")
            frame_placeholder = st.empty()
            
            process_video = st.button("🎬 Process Video")
            
            if process_video:
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 5th frame for speed
                    if frame_count % 5 == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = process_frame(frame_rgb)
                        frame_placeholder.image(result, channels="RGB", use_container_width=True)
                    
                    frame_count += 1
                
                cap.release()
                st.success(f"✅ Processed {frame_count} frames!")
        else:
            st.info("👆 Upload a video to get started")

# Webcam (Local Only)
else:
    with col1:
        run_detection = st.checkbox("Start Webcam", value=False)
        frame_placeholder = st.empty()
        
        if run_detection:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("⚠️ Cannot access webcam. This feature only works when running locally!")
                st.info("Try running: `streamlit run app.py` on your computer")
            else:
                while run_detection:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame")
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = process_frame(frame_rgb)
                    frame_placeholder.image(result, channels="RGB", use_container_width=True)
                
                cap.release()
        else:
            st.info("👈 Check 'Start Webcam' to begin (works only locally)")
