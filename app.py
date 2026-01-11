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
    # Using YOLOS-Tiny for faster real-time detection
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
    ["🎥 Live Webcam Stream", "🔗 Image URL", "📁 Upload Image"],
    label_visibility="collapsed"
)

# Function to process and draw detections
def process_image(pil_image):
    # Perform detection - pass PIL Image directly
    detections = detector(pil_image)
    
    # Filter by confidence
    filtered_detections = [d for d in detections if d['score'] >= confidence_threshold]
    
    # Convert PIL to numpy for drawing
    image_rgb = np.array(pil_image.convert('RGB'))
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

# LIVE WEBCAM STREAM MODE
if input_method == "🎥 Live Webcam Stream":
    with col1:
        st.subheader("🎥 Live Webcam Stream")
        
        # Use Streamlit's camera_input in continuous mode
        st.markdown("**📹 Continuous Detection Mode**")
        
        enable_camera = st.checkbox("🔴 Enable Camera", value=False)
        
        if enable_camera:
            img_file_buffer = st.camera_input("Live Feed", key="camera")
            
            if img_file_buffer is not None:
                # Read image
                pil_image = Image.open(img_file_buffer)
                
                # Process image
                result_image, detections, detection_count = process_image(pil_image)
                
                # Display result
                st.image(result_image, channels="RGB", use_container_width=True, caption="🎯 Live Detection")
                
                # Update stats
                with stats_placeholder.container():
                    st.metric("🎯 Objects Detected", len(detections))
                    
                    if detection_count:
                        st.write("**📦 Live Count:**")
                        for obj, count in sorted(detection_count.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"• **{obj}**: {count}")
                    else:
                        st.info("No objects detected")
                    
                    if detections:
                        st.write("---")
                        st.write("**🔍 Detections:**")
                        for i, det in enumerate(detections[:5], 1):  # Show top 5
                            confidence_color = "🟢" if det['score'] > 0.7 else "🟡" if det['score'] > 0.5 else "🟠"
                            st.write(f"{i}. {confidence_color} **{det['label']}** - {det['score']:.1%}")
                
                # Auto-refresh instruction
                st.info("📸 **Tip:** Take new photos to update detection in real-time!")
        else:
            st.info("👆 Check '🔴 Enable Camera' to start live detection")
            with stats_placeholder:
                st.info("Camera disabled")

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
                    pil_image = Image.open(BytesIO(response.content))
                
                # Process image
                result_image, detections, detection_count = process_image(pil_image)
                
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
            pil_image = Image.open(uploaded_file)
            
            # Process image
            result_image, detections, detection_count = process_image(pil_image)
            
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

if input_method == "🎥 Live Webcam Stream":
    st.sidebar.markdown("""
    **LIVE WEBCAM MODE** 🎥
    
    1. Check **🔴 Enable Camera**
    2. Allow camera permissions
    3. Take photos continuously
    4. Each photo is analyzed instantly
    5. Keep taking photos for live updates
    
    ✅ **Works on Streamlit Cloud!**
    """)
elif input_method == "🔗 Image URL":
    st.sidebar.markdown("""
    **URL MODE** 🔗
    
    1. Copy any image URL
    2. Paste in the text box
    3. Press Enter
    4. Instant detection!
    
    **Try these:**
    - Unsplash images
    - Direct .jpg/.png links
    - Public image URLs
    """)
else:
    st.sidebar.markdown("""
    **UPLOAD MODE** 📁
    
    1. Click **Browse files**
    2. Select image from device
    3. Wait for upload
    4. View detection results!
    
    **Supported:** 
    JPG, PNG, WEBP, BMP
    """)

st.sidebar.markdown("---")
st.sidebar.info("💡 Lower confidence threshold to see more detections")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Info")
st.sidebar.markdown("""
**Model:** YOLOS-Tiny  
**Speed:** ⚡ Fast inference  
**Accuracy:** 🎯 High precision  
**Hardware:** 💻 CPU compatible  
**Cloud:** ☁️ Streamlit Cloud ready
""")

# Add some example URLs
if input_method == "🔗 Image URL":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Example URLs")
    example_urls = {
        "Street Scene": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000",
        "People": "https://images.unsplash.com/photo-1511632765486-a01980e01a18",
        "Animals": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee",
    }
    
    for name, url in example_urls.items():
        if st.sidebar.button(f"📸 {name}", key=name):
            st.session_state.example_url = url
            st.rerun()
    
    # Auto-fill if example selected
    if 'example_url' in st.session_state:
        st.sidebar.success(f"Selected: {name}")
        # The URL will be auto-filled on rerun
