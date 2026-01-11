import streamlit as st
from PIL import Image
import numpy as np
from transformers import pipeline
import torch
import cv2
import requests
from io import BytesIO
import time

# Page config
st.set_page_config(page_title="Real-Time Object Detection", layout="wide")

# Cache the model to avoid reloading
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    # Using YOLOv5 for faster real-time detection
    return pipeline("object-detection", model="hustvl/yolos-tiny", device=device)

# Load model
with st.spinner("🔄 Loading AI model..."):
    detector = load_model()

# Title
st.title("🎥 Live Object Detection")
st.markdown("Real-time AI detection using **YOLOS-Tiny** model")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# Input method selection
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input Source")
input_method = st.sidebar.radio(
    "Choose input:",
    ["🔴 Live Webcam", "🔗 Image URL", "📁 Upload Image"],
    label_visibility="collapsed"
)

# Function to process and draw detections
def process_image(image_rgb):
    # Perform detection
    detections = detector(image_rgb)
    
    # Filter by confidence
    filtered_detections = [d for d in detections if d['score'] >= confidence_threshold]
    
    # Draw bounding boxes with enhanced visibility
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
        
        # Draw thick rectangle for better visibility
        cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        
        # Prepare text with better formatting
        text = f"{label}: {score:.0%}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.9
        font_thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw black background rectangle for text (with padding)
        padding = 10
        cv2.rectangle(frame_copy, 
                     (xmin - 2, ymin - text_height - padding - baseline),
                     (xmin + text_width + padding, ymin),
                     (0, 0, 0), -1)
        
        # Draw green border around text background
        cv2.rectangle(frame_copy,
                     (xmin - 2, ymin - text_height - padding - baseline),
                     (xmin + text_width + padding, ymin),
                     (0, 255, 0), 3)
        
        # Draw white text on black background
        cv2.putText(frame_copy, text, 
                   (xmin + 4, ymin - padding + 2), 
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return frame_copy, filtered_detections, detection_count

# Main content
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Live Stats")
    stats_placeholder = st.empty()
    fps_placeholder = st.empty()

# LIVE WEBCAM MODE
if input_method == "🔴 Live Webcam":
    with col1:
        st.subheader("🔴 Live Detection")
        
        # Create placeholders
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Control buttons
        col_a, col_b = st.columns(2)
        with col_a:
            start_button = st.button("▶️ Start Live Detection", use_container_width=True)
        with col_b:
            stop_button = st.button("⏹️ Stop", use_container_width=True)
        
        # Session state for control
        if 'running' not in st.session_state:
            st.session_state.running = False
        
        if start_button:
            st.session_state.running = True
        if stop_button:
            st.session_state.running = False
        
        # Live detection loop
        if st.session_state.running:
            status_placeholder.success("🔴 LIVE - Detection running...")
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                status_placeholder.error("❌ Cannot access webcam. Please allow camera permissions!")
                st.session_state.running = False
            else:
                frame_count = 0
                start_time = time.time()
                
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.error("❌ Failed to grab frame")
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    result_image, detections, detection_count = process_image(frame_rgb)
                    
                    # Display frame
                    frame_placeholder.image(result_image, channels="RGB", use_container_width=True)
                    
                    # Calculate FPS
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    # Update stats
                    with stats_placeholder.container():
                        st.metric("🎯 Objects Detected", len(detections))
                        
                        if detection_count:
                            st.write("**📦 Live Count:**")
                            for obj, count in sorted(detection_count.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"• **{obj}**: {count}")
                        else:
                            st.info("No objects detected")
                    
                    with fps_placeholder:
                        st.metric("⚡ FPS", f"{fps:.1f}")
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.03)
                
                cap.release()
                status_placeholder.info("⏸️ Detection stopped")
        else:
            frame_placeholder.info("👆 Click '▶️ Start Live Detection' to begin")
            with stats_placeholder:
                st.info("Waiting to start...")

# URL INPUT MODE
elif input_method == "🔗 Image URL":
    with col1:
        st.subheader("🔗 Image from URL")
        image_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        
        if image_url:
            try:
                with st.spinner("⏳ Downloading image..."):
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    image_rgb = np.array(image.convert('RGB'))
                
                # Process image
                result_image, detections, detection_count = process_image(image_rgb)
                
                # Display result
                st.image(result_image, channels="RGB", use_container_width=True, caption="🎯 Detection Results")
                
                # Display stats
                with stats_placeholder.container():
                    st.metric("🎯 Total Objects", len(detections))
                    
                    if detection_count:
                        st.write("**📦 Object Count:**")
                        for obj, count in sorted(detection_count.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"• **{obj}**: {count}")
                    else:
                        st.info("No objects detected")
                    
                    if detections:
                        st.write("---")
                        st.write("**🔍 All Detections:**")
                        for i, det in enumerate(detections, 1):
                            confidence_color = "🟢" if det['score'] > 0.7 else "🟡" if det['score'] > 0.5 else "🟠"
                            st.write(f"{i}. {confidence_color} **{det['label']}** - {det['score']:.1%}")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error loading image: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        else:
            st.info("👆 Enter an image URL above to detect objects")
            with stats_placeholder:
                st.info("Waiting for URL...")

# FILE UPLOAD MODE
else:
    with col1:
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_rgb = np.array(image.convert('RGB'))
            
            # Process image
            result_image, detections, detection_count = process_image(image_rgb)
            
            # Display result
            st.image(result_image, channels="RGB", use_container_width=True, caption="🎯 Detection Results")
            
            # Display stats
            with stats_placeholder.container():
                st.metric("🎯 Total Objects", len(detections))
                
                if detection_count:
                    st.write("**📦 Object Count:**")
                    for obj, count in sorted(detection_count.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"• **{obj}**: {count}")
                else:
                    st.info("No objects detected")
                
                if detections:
                    st.write("---")
                    st.write("**🔍 All Detections:**")
                    for i, det in enumerate(detections, 1):
                        confidence_color = "🟢" if det['score'] > 0.7 else "🟡" if det['score'] > 0.5 else "🟠"
                        st.write(f"{i}. {confidence_color} **{det['label']}** - {det['score']:.1%}")
        else:
            st.info("👆 Upload an image to detect objects")
            with stats_placeholder:
                st.info("Waiting for upload...")

# Sidebar instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 How to Use")

if input_method == "🔴 Live Webcam":
    st.sidebar.markdown("""
    **LIVE MODE** ⚡
    
    1. Click **▶️ Start Live Detection**
    2. Allow camera permissions
    3. View real-time detection
    4. Click **⏹️ Stop** when done
    
    ⚠️ **Note:** Live mode works best locally.
    On Streamlit Cloud, use URL or Upload.
    """)
elif input_method == "🔗 Image URL":
    st.sidebar.markdown("""
    **URL MODE** 🔗
    
    1. Copy any image URL
    2. Paste in the text box
    3. Press Enter
    4. Instant detection!
    
    **Example:**
    ```
    https://images.unsplash.com/
    photo-xyz.jpg
    ```
    """)
else:
    st.sidebar.markdown("""
    **UPLOAD MODE** 📁
    
    1. Click **Browse files**
    2. Select image from device
    3. Wait for upload
    4. View detection results!
    
    **Supported:** JPG, PNG, WEBP, BMP
    """)

st.sidebar.markdown("---")
st.sidebar.info("💡 Lower confidence threshold to see more detections")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Info")
st.sidebar.markdown("""
**Model:** YOLOS-Tiny  
**Speed:** ⚡ Optimized for real-time  
**Accuracy:** 🎯 High precision  
**Hardware:** 💻 CPU compatible
""")
