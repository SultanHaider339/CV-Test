import streamlit as st
from PIL import Image
import numpy as np
from transformers import pipeline
import torch
import cv2

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

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Webcam Feed")
    # Camera input - works on Streamlit Cloud!
    camera_photo = st.camera_input("Take a photo")

with col2:
    st.subheader("Detection Stats")
    stats_placeholder = st.empty()

# Process camera image
if camera_photo is not None:
    # Read image
    image = Image.open(camera_photo)
    image_rgb = np.array(image.convert('RGB'))
    
    # Perform detection
    with st.spinner("Detecting objects..."):
        detections = detector(image)
    
    # Filter by confidence
    filtered_detections = [d for d in detections if d['score'] >= confidence_threshold]
    
    # Draw bounding boxes
    frame_copy = image_rgb.copy()
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
        cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        
        # Put text with background
        text = f"{label}: {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_copy, (xmin, ymin - text_height - 10), 
                     (xmin + text_width, ymin), (0, 255, 0), -1)
        cv2.putText(frame_copy, text, (xmin, ymin - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Display result
    with col1:
        st.image(frame_copy, channels="RGB", use_container_width=True, caption="Detected Objects")
    
    # Display stats
    with stats_placeholder.container():
        st.metric("Total Objects", len(filtered_detections))
        if detection_count:
            st.write("**Detected Objects:**")
            for obj, count in sorted(detection_count.items(), key=lambda x: x[1], reverse=True):
                st.write(f"• **{obj}**: {count}")
        else:
            st.info("No objects detected")
        
        # Show confidence scores
        if filtered_detections:
            st.write("---")
            st.write("**All Detections:**")
            for i, det in enumerate(filtered_detections, 1):
                st.write(f"{i}. {det['label']} - {det['score']:.1%}")

else:
    with col1:
        st.info("👆 Click 'Take a photo' to capture an image from your webcam")
    with stats_placeholder:
        st.info("Waiting for image...")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 How to Use")
st.sidebar.markdown("""
1. Click **'Take a photo'** button
2. Allow camera access in your browser
3. Point camera at objects
4. Click **'Take Photo'** to capture
5. View detected objects instantly!
6. Take another photo to detect again
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Adjust confidence threshold to filter detections")
