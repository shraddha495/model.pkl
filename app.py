import time
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_lottie import st_lottie

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Model Deployment App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CUSTOM CSS & ANIMATIONS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Background & Font Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header Container */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Custom Card Containers */
    .card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
    }
    
    /* Result Display */
    .prediction-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Helper function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_ai = load_lottieurl(
    "https://assets5.lottiefiles.com/packages/lf20_m64ro9yb.json"
)


# Load the model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model.pkl: {e}")
        return None


model = load_model()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    if lottie_ai:
        st_lottie(lottie_ai, height=180, key="ai_anim")
    st.title("⚙️ Configuration")
    st.markdown("---")
    st.info(
        "Adjust input variables on the right pane and click **Run Prediction** to test model outputs."
    )
    st.markdown("### 📊 Model Info")
    st.text("File: model.pkl")
    st.text("Status: Loaded Successfully" if model else "Status: Failed to Load")

# -----------------------------------------------------------------------------
# MAIN CONTENT LAYOUT
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1 style="margin: 0; color: #8b5cf6;">🚀 Machine Learning Inference Engine</h1>
        <p style="color: #94a3b8; margin-top: 8px;">Interactive Web Deployment Platform</p>
    </div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📥 Input Features")

    with st.form("prediction_form"):
        # DEFINE YOUR MODEL INPUTS HERE
        f_col1, f_col2 = st.columns(2)

        with f_col1:
            feature_1 = st.number_input(
                "Feature 1 (Numeric)", value=0.0, step=0.1
            )
            feature_2 = st.slider(
                "Feature 2 (Slider)",
                min_value=0,
                max_value=100,
                value=50,
            )

        with f_col2:
            feature_3 = st.selectbox(
                "Feature 3 (Categorical)", options=[0, 1, 2]
            )
            feature_4 = st.number_input("Feature 4 (Numeric)", value=10.0)

        submit_btn = st.form_submit_button("⚡ Run Prediction")

with col2:
    st.markdown("### 🎯 Model Output")

    if submit_btn:
        if model is None:
            st.error("Model file `model.pkl` could not be loaded.")
        else:
            # Display processing spinner
            with st.spinner("Processing feature vector..."):
                time.sleep(0.8)  # Visual effect delay

            # Format features for prediction
            input_data = np.array(
                [[feature_1, feature_2, feature_3, feature_4]]
            )

            try:
                prediction = model.predict(input_data)[0]

                # --- PREDICT CLICK EFFECTS ---
                st.balloons()
                st.snow()

                # Result Output Display
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        Result: {prediction}
                    </div>
                """,
                    unsafe_allow_html=True,
                )

                # Optional: Probability readout if classifier supports it
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_data)[0]
                    st.write("")
                    st.write("**Prediction Confidence:**")
                    st.progress(float(np.max(probs)))

            except Exception as e:
                st.error(f"Inference failed: {e}")
    else:
        st.info("Submit feature values on the left to display predictions.")
