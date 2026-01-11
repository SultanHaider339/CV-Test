import streamlit as st
from PIL import Image
import numpy as np
from transformers import pipeline
import torch
import cv2
import requests
from io import BytesIO

# Page config
st.set_page_config(page_title="Real-Time Object Detection", layout="wide")

# Cache the model to avoid reloading
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("object-detection", model="hustvl/yolos-tiny", device=device)

# Load model
with st.spinner("🔄 Loading AI model..."):
    detector = load_model()

# Title
st.title("🎥 Real-Time Object Detection")
st.markdown("Live AI detection using **YOLOS-Tiny** model")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# Input method selection
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input Source")
input_method = st.sidebar.radio(
    "Choose input:",
    ["🎥 Live Webcam", "🔗 Image URL", "📁 Upload Image"],
    label_visibility="collapsed"
)

# Function to process and draw detections
def process_image(pil_image):
    # Perform detection
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
        
        # Draw thick rectangle
        cv2.rectangle(frame_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)
        
        # Prepare text
        text = f"{label}: {score:.0%}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.9
        font_thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw text background
        padding = 10
        cv2.rectangle(frame_copy, 
                     (xmin - 2, ymin - text_height - padding - baseline),
                     (xmin + text_width + padding, ymin),
                     (0, 0, 0), -1)
        
        # Draw border
        cv2.rectangle(frame_copy,
                     (xmin - 2, ymin - text_height - padding - baseline),
                     (xmin + text_width + padding, ymin),
                     (0, 255, 0), 3)
        
        # Draw text
        cv2.putText(frame_copy, text, 
                   (xmin + 4, ymin - padding + 2), 
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return frame_copy, filtered_detections, detection_count

# Main content
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Detection Stats")
    stats_placeholder = st.empty()

# LIVE WEBCAM MODE - Auto-updating
if input_method == "🎥 Live Webcam":
    with col1:
        st.subheader("🎥 Live Webcam Detection")
        
        # Webcam widget that auto-updates
        img_file_buffer = st.camera_input("Live Feed")
        
        if img_file_buffer is not None:
            # Read and process image directly without downloading
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
                    for i, det in enumerate(detections[:5], 1):
                        confidence_color = "🟢" if det['score'] > 0.7 else "🟡" if det['score'] > 0.5 else "🟠"
                        st.write(f"{i}. {confidence_color} **{det['label']}** - {det['score']:.1%}")
            
            # Add auto-refresh button
            st.info("📸 **Live Mode Active:** Take new photos to update detection continuously!")
        else:
            st.info("👆 Click 'Take a photo' to start live detection")
            with stats_placeholder:
                st.info("Waiting for camera input...")

# URL INPUT MODE - Direct processing without download
elif input_method == "🔗 Image URL":
    with col1:
        st.subheader("🔗 Detect from Image URL")
        
        # URL input with session state
        if 'url_input' not in st.session_state:
            st.session_state.url_input = ''
        
        image_url = st.text_input(
            "Enter image URL:", 
            value=st.session_state.url_input,
            placeholder="https://example.com/image.jpg",
            key="url_field"
        )
        
        if image_url:
            try:
                # Stream image directly without downloading to disk
                with st.spinner("⏳ Loading image..."):
                    response = requests.get(image_url, timeout=10, stream=True)
                    response.raise_for_status()
                    
                    # Process directly from stream
                    pil_image = Image.open(BytesIO(response.content))
                
                # Process image in memory
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
                
                st.success("✅ Image processed directly from URL - no download required!")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error loading image from URL: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        else:
            st.info("👆 Enter an image URL above to detect objects instantly")
            with stats_placeholder:
                st.info("Waiting for URL...")

# FILE UPLOAD MODE - Direct processing from memory
else:
    with col1:
        st.subheader("📁 Upload and Detect")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
        
        if uploaded_file is not None:
            # Process directly from upload buffer without saving
            pil_image = Image.open(uploaded_file)
            
            # Process image in memory
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
            
            st.success("✅ Image processed directly from upload - no file saved!")
        else:
            st.info("👆 Upload an image to detect objects instantly")
            with stats_placeholder:
                st.info("Waiting for upload...")

# Sidebar instructions
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 How to Use")

if input_method == "🎥 Live Webcam":
    st.sidebar.markdown("""
    **LIVE WEBCAM MODE** 🎥
    
    ✨ **Real-time detection:**
    1. Click **'Take a photo'**
    2. Allow camera permissions
    3. Image is analyzed instantly
    4. Keep taking photos for continuous updates
    
    🚀 **No downloads required!**
    Everything happens in browser memory
    
    ✅ Works on Streamlit Cloud
    """)
elif input_method == "🔗 Image URL":
    st.sidebar.markdown("""
    **URL MODE** 🔗
    
    ✨ **Instant detection:**
    1. Paste any image URL
    2. Press Enter
    3. Image streams directly to AI
    4. Results appear instantly
    
    🚀 **No downloads to disk!**
    Image processed in memory only
    
    **Try example URLs below** ⬇️
    """)
else:
    st.sidebar.markdown("""
    **UPLOAD MODE** 📁
    
    ✨ **Quick detection:**
    1. Click **'Browse files'**
    2. Select image from device
    3. Processing starts immediately
    4. View instant results
    
    🚀 **No file saving!**
    Upload processed in memory
    
    **Supported:** JPG, PNG, WEBP, BMP
    """)

st.sidebar.markdown("---")
st.sidebar.info("💡 Adjust confidence threshold to filter detections")

# Example URLs for quick testing
if input_method == "🔗 Image URL":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Quick Test URLs")
    
    example_urls = {
        "🏙️ Street Scene": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
        "👥 People": "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=800",
        "🐕 Animals": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=800",
        "🚗 Traffic": "https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800",
    }
    
    for name, url in example_urls.items():
        if st.sidebar.button(name, use_container_width=True):
            st.session_state.url_input = url
            st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Info")
st.sidebar.markdown("""
**Model:** YOLOS-Tiny  
**Processing:** 🚀 Real-time in-memory  
**Storage:** 💾 Zero disk usage  
**Speed:** ⚡ Instant detection  
**Cloud:** ☁️ Fully cloud-compatible
""")

st.sidebar.markdown("---")
st.sidebar.success("🎯 All processing happens in memory - no downloads!")
