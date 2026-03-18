import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import pickle
import os

# -----------------------------------------------------------
# 🌟 Page Config
# -----------------------------------------------------------
st.set_page_config(
    page_title="Alzheimer's Detection | AI Diagnostics", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------
# 🎨 Modern UI Styling (Glassmorphism + Premium Aesthetics)
# -----------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    /* Global styles */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Glass Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin: 2rem auto;
        max-width: 900px;
    }

    /* Header Styling */
    .hero-title {
        background: linear-gradient(to right, #60a5fa, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    /* Prediction Result Styles */
    .prediction-card {
        background: rgba(96, 165, 250, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1rem;
        background: #2563eb;
        color: white;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
        border: none;
        color: white;
    }

    /* File Uploader Customization */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.05);
    }

    /* Info Cards */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 3rem;
    }
    .info-item {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧠 Load Model & Config
# -----------------------------------------------------------
@st.cache_resource
def load_alzheimer_model():
    model_path_h5 = "alzheimer_cnn_model.h5"
    model_path_pkl = "alzheimer_model.pkl"
    
    try:
        if os.path.exists(model_path_h5):
            return tf.keras.models.load_model(model_path_h5)
        elif os.path.exists(model_path_pkl):
            with open(model_path_pkl, "rb") as f:
                return pickle.load(f)
        else:
            st.error("Model file not found! Please ensure 'alzheimer_model.pkl' or 'alzheimer_cnn_model.h5' exists.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_alzheimer_model()
class_names = ['Mild Demented', 'Moderate Demented', 'Non-Demented', 'Very Mild Demented']

# -----------------------------------------------------------
# 🏗️ Content Layout
# -----------------------------------------------------------

# Hero Section
st.markdown('<h1 class="hero-title">NeuroScan AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Advanced Neural Network Analysis for Early Alzheimer\'s Detection</p>', unsafe_allow_html=True)

# Main content
main_col_1, main_col_2, main_col_3 = st.columns([1, 4, 1])

with main_col_2:
    # Upload Section
    st.write("### 📂 Upload MRI Image")
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        # Image Processing
        image = Image.open(uploaded_file).convert("RGB")
        
        # UI Columns for Preview and Results
        preview_col, result_col = st.columns([1, 1], gap="large")
        
        with preview_col:
            st.write("#### 🎥 Preview")
            st.image(image, use_container_width=True)
        
        with result_col:
            st.write("#### ⚡ Analysis Control")
            if st.button("🚀 Run AI Diagnostic"):
                if model is None:
                    st.error("Diagnostic engine failed to initialize. Please check model files.")
                else:
                    with st.spinner("Processing through neural layers..."):
                        # Preprocessing
                        img = image.resize((128, 128))
                        img_array = img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Prediction
                        prediction = model.predict(img_array)
                        predicted_idx = np.argmax(prediction)
                        predicted_class = class_names[predicted_idx]
                        confidence = np.max(prediction) * 100
                        
                        # Results Display
                        st.markdown(f"""
                        <div class="prediction-card">
                            <span class="status-badge">Diagnostic Complete</span>
                            <h2 style="margin:0; color:#60a5fa;">{predicted_class}</h2>
                            <p style="color:#94a3b8; margin-top:0.5rem;">Confidence Level: <b>{confidence:.2f}%</b></p>
                            <div style="background:#1e293b; height:8px; border-radius:4px; margin-top:1rem;">
                                <div style="background:linear-gradient(to right, #3b82f6, #a855f7); width:{confidence}%; height:100%; border-radius:4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if predicted_class != "Non-Demented":
                            st.warning("⚠️ **Note:** Please consult with a medical professional.")
                        else:
                            st.success("✅ Analysis suggests no significant patterns of dementia detected.")

# Footer
st.markdown("---")
st.markdown('<p style="text-align:center; color:#475569; font-size:0.8rem;">© 2024 AI Healthcare Diagnostics • Engineered by Arya and Sahil • For research purposes only</p>', unsafe_allow_html=True)
