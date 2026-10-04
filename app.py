import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Academic Score Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for Dark Sky Blue Theme, Sparkling Golden Stars & Modern Styling
st.markdown(
    """
    <style>
    /* Main Background: Dark Sky Blue Gradient */
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0b192c 50%, #1e293b 100%);
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
        position: relative;
        overflow-x: hidden;
    }

    /* Sparkling Golden Stars Background Effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        background-image: 
            radial-gradient(3px 3px at 20px 30px, #ffd700, rgba(0,0,0,0)),
            radial-gradient(4px 4px at 40px 70px, #ffdf00, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 50px 160px, #fff, rgba(0,0,0,0)),
            radial-gradient(5px 5px at 90px 40px, #ffd700, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 130px 80px, #ffea79, rgba(0,0,0,0)),
            radial-gradient(4px 4px at 160px 120px, #ffd700, rgba(0,0,0,0)),
            radial-gradient(6px 6px at 200px 200px, #ffcc00, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 250px 50px, #fff, rgba(0,0,0,0)),
            radial-gradient(5px 5px at 300px 180px, #ffd700, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 350px 90px, #ffdf00, rgba(0,0,0,0)),
            radial-gradient(4px 4px at 400px 250px, #ffd700, rgba(0,0,0,0)),
            radial-gradient(6px 6px at 450px 150px, #ffaa00, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 500px 300px, #ffd700, rgba(0,0,0,0)),
            radial-gradient(5px 5px at 600px 100px, #fff, rgba(0,0,0,0)),
            radial-gradient(4px 4px at 700px 220px, #ffd700, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 700px 500px;
        opacity: 0.85;
        animation: sparkle 4s ease-in-out infinite alternate;
        z-index: 1;
    }

    @keyframes sparkle {
        0% { transform: scale(1); opacity: 0.6; }
        50% { transform: scale(1.03); opacity: 1; }
        100% { transform: scale(1); opacity: 0.7; }
    }

    /* Ensure Streamlit content stays above the background */
    .main .block-container {
        position: relative;
        z-index: 2;
    }

    /* Glassmorphism Card Container Style */
    div.stContainer, div[data-testid="stVerticalBlock"] > div > div.stForm {
        background-color: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 215, 0, 0.2);
        padding: 24px;
        border-radius: 14px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Input Labels and Text Adjustments */
    label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    /* Prediction Button Customization with Gold/Purple Glow */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d97706 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 10px;
        border: 1px solid rgba(255, 215, 0, 0.3);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(255, 215, 0, 0.5);
        border-color: #ffd700;
    }

    /* Result Box Styling */
    .result-box {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.9) 0%, rgba(4, 47, 46, 0.9) 100%);
        border: 1px solid #34d399;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.25);
    }
    .result-text {
        color: #34d399;
        font-size: 26px;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(52, 211, 153, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load the Pickled Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# Application Title & Subtitle
st.title("🎓 Academic Performance Predictor")
st.caption(
    "Input student subject scores below to generate total predictions using your KNN model."
)

st.markdown("---")

# Input Form Container
with st.container():
    st.write("### 📝 Enter Subject Scores")

    col1, col2 = st.columns(2)

    with col1:
        hindi = st.number_input(
            "Hindi", min_value=0, max_value=100, value=75, step=1
        )
        english = st.number_input(
            "English", min_value=0, max_value=100, value=80, step=1
        )
        science = st.number_input(
            "Science", min_value=0, max_value=100, value=85, step=1
        )

    with col2:
        maths = st.number_input(
            "Maths", min_value=0, max_value=100, value=90, step=1
        )
        history = st.number_input(
            "History", min_value=0, max_value=100, value=70, step=1
        )
        geography = st.number_input(
            "Geography", min_value=0, max_value=100, value=75, step=1
        )

    # Automatic Total Calculation as the 7th Feature
    calculated_total = hindi + english + science + maths + history + geography
    st.info(f"**Calculated Total Score:** {calculated_total} / 600")

# Prediction Action
if st.button("🔮 Run Prediction"):
    # Trigger Balloons for Visual Effect
    st.balloons()

    # Format features matching model schema: ['Hindi', 'English', 'Science', 'Maths', 'History', 'Geography', 'Total']
    features = np.array(
        [[hindi, english, science, maths, history, geography, calculated_total]]
    )

    # Perform Prediction
    prediction = model.predict(features)[0]

    # Display Pretty Result Output
    st.markdown(
        f"""
        <div class="result-box">
            <span style="color: #cbd5e1; font-size: 16px;">Predicted Outcome / Class:</span>
            <div class="result-text">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.snow()
