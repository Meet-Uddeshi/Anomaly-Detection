import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import time
import os

st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .anomaly-detected {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .normal-detected {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .stProgress .st-bo {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def fix_depthwise_conv2d():
    try:
        custom_objects = {}
        original_depthwise_conv2d = keras.layers.DepthwiseConv2D
        class FixedDepthwiseConv2D(original_depthwise_conv2d):
            def __init__(self, *args, **kwargs):
                if 'groups' in kwargs:
                    kwargs.pop('groups')
                super().__init__(*args, **kwargs)
        custom_objects['DepthwiseConv2D'] = FixedDepthwiseConv2D
        return custom_objects
    except Exception:
        return {}

@st.cache_resource
def load_model():
    model_path = 'keras_model.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it's in the same directory as this script.")
        return None
    try:
        custom_objects = fix_depthwise_conv2d()
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success("Model loaded successfully with custom objects!")
        return model
    except Exception as e1:
        st.warning(f"First attempt failed: {str(e1)}")
        try:
            model = keras.models.load_model(model_path, compile=False)
            st.success("Model loaded successfully without compilation!")
            return model
        except Exception as e2:
            st.warning(f"Second attempt failed: {str(e2)}")
            try:
                model = tf.saved_model.load(model_path)
                st.success("Model loaded as TensorFlow SavedModel!")
                return model
            except Exception as e3:
                st.error(f"All loading methods failed:")
                st.error(f"Method 1 (custom objects): {str(e1)}")
                st.error(f"Method 2 (no compile): {str(e2)}")
                st.error(f"Method 3 (SavedModel): {str(e3)}")
                return None

def preprocess_image(image, target_size=(224, 224)):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale with channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_resized = cv2.resize(image, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        return image_batch
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

def predict_anomaly(model, image, threshold=0.5):
    try:
        if model is None:
            return None, None    
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None
        if hasattr(model, 'predict'):
            prediction = model.predict(processed_image, verbose=0)
        else:
            prediction = model(processed_image)
            if hasattr(prediction, 'numpy'):
                prediction = prediction.numpy()
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if prediction.ndim > 1:
            confidence = float(prediction[0][0]) if prediction.shape[1] == 1 else float(np.max(prediction[0]))
        else:
            confidence = float(prediction[0])
        is_anomaly = confidence > threshold
        return is_anomaly, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def capture_from_camera():
    """Capture image from camera with error handling"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera. Please check if your camera is available.")
            return None
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            st.error("Failed to capture image from camera.")
            return None
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">Anomaly Detection </h1>', unsafe_allow_html=True)
    with st.spinner("Loading model..."):
        model = load_model()
    if model is None:
        st.error("Cannot proceed without a valid model. Please fix the model loading issue.")
        st.stop()
    st.sidebar.header("Configuration")
    threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.01)
    st.sidebar.subheader("Model Status")
    if hasattr(model, 'input_shape'):
        st.sidebar.success(f"✅ Model loaded\nInput shape: {model.input_shape}")
    else:
        st.sidebar.success("✅ Model loaded (SavedModel format)")
    st.sidebar.subheader("Detection Method")
    feature = st.sidebar.radio(
        "Choose method:",
        ["Upload Image", "Camera Capture"]
    )
    col1, col2 = st.columns([2, 1])
    if feature == "Upload Image":
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("📁 Upload Image Analysis")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload an image file for anomaly detection"
            )
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    st.write(f"**Image Size:** {image.size}")
                    st.write(f"**Image Mode:** {image.mode}")
                    if st.button("🔍 Analyze Image", type="primary"):
                        with st.spinner("Analyzing image..."):
                            is_anomaly, confidence = predict_anomaly(model, image, threshold)
                            if is_anomaly is not None:
                                st.session_state.last_result = {
                                    'image': np.array(image),
                                    'is_anomaly': is_anomaly,
                                    'confidence': confidence,
                                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                                }
                                st.success("✅ Analysis complete!")
                            else:
                                st.error("❌ Analysis failed!")
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:  
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("📸 Camera Capture Analysis")
            st.info("Click the button below to capture an image from your camera")
            if st.button("📷 Capture & Analyze", type="primary"):
                with st.spinner("Capturing image..."):
                    captured_image = capture_from_camera()
                    if captured_image is not None:
                        st.image(captured_image, caption="Captured Image", use_column_width=True)
                        with st.spinner("Analyzing captured image...")
                            is_anomaly, confidence = predict_anomaly(model, captured_image, threshold)
                            if is_anomaly is not None:
                                st.session_state.last_result = {
                                    'image': captured_image,
                                    'is_anomaly': is_anomaly,
                                    'confidence': confidence,
                                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                                }
                                st.success("✅ Capture and analysis complete!")
                            else:
                                st.error("❌ Analysis failed!")
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.subheader("📊 Detection Results")
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            if result['is_anomaly']:
                st.markdown(
                    f'<div class="prediction-box anomaly-detected">'
                    f'<h3>⚠️ ANOMALY DETECTED</h3>'
                    f'<p><strong>Confidence:</strong> {result["confidence"]:.2%}</p>'
                    f'<p><strong>Status:</strong> Anomalous</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-box normal-detected">'
                    f'<h3>✅ NORMAL</h3>'
                    f'<p><strong>Confidence:</strong> {result["confidence"]:.2%}</p>'
                    f'<p><strong>Status:</strong> Normal</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.subheader("📈 Confidence Score")
            st.progress(result['confidence'])
            st.write(f"Score: {result['confidence']:.4f}")
            st.subheader("📋 Analysis Details")
            details_col1, details_col2 = st.columns(2)
            with details_col1:
                st.metric("Threshold", f"{threshold:.2f}")
                st.metric("Score", f"{result['confidence']:.3f}")
            with details_col2:
                st.write("**Timestamp:**")
                st.write(result['timestamp'])
                st.write("**Result:**")
                st.write("🔴 Anomaly" if result['is_anomaly'] else "🟢 Normal")       
        else:
            st.info("No analysis performed yet. Upload an image or capture from camera to get started!")
            
            # Help section
            st.subheader("💡 How to Use")
            st.write("""
            1. **Upload Image**: Choose a PNG/JPG file from your computer
            2. **Camera Capture**: Take a photo using your device's camera
            3. **Adjust Threshold**: Use the slider in the sidebar to fine-tune sensitivity
            4. **View Results**: Check this panel for detection results and confidence scores
            """)
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        f"Anomaly Detection System v2.0 | TensorFlow {tf.__version__} | "
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()