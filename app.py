import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Academic Score Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for Modern Styling, Sky-Blue Dark Background, and Sparkling Golden Stars
st.markdown(
    """
    <style>
    /* Animated Sky-Blue Dark Background with Sparkling Golden Stars */
    .stApp {
        background: radial-gradient(circle at center, #0f172a 0%, #090d16 100%);
        font-family: 'Inter', sans-serif;
        position: relative;
        overflow-x: hidden;
    }

    /* Starry Background Effect */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2.5px 2.5px at 20px 30px, #ffd700, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 40px 70px, #ffdf00, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 50px 160px, #ffe066, rgba(0,0,0,0)),
            radial-gradient(3.5px 3.5px at 90px 40px, #ffcc00, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 130px 80px, #ffd700, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 160px 120px, #ffea80, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 250px 250px;
        opacity: 0.65;
        animation: starSparkle 5s ease-in-out infinite alternate;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes starSparkle {
        0% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.05); opacity: 0.85; }
        100% { transform: scale(1); opacity: 0.5; }
    }

    /* Ensure content stays above background elements */
    .main > div {
        position: relative;
        z-index: 1;
    }

    /* Card Container Style */
    .stCard, div[data-testid="stVerticalBlock"] > div.element-container {
        position: relative;
        z-index: 1;
    }

    /* App Header Styling for Dark Theme Visibility */
    h1, h2, h3, h4, h5, h6, label, .stMarkdown p {
        color: #f1f5f9 !important;
    }

    /* Input Fields Styling */
    .stNumberInput input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Info Box Customization */
    .stInfo {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: #38bdf8 !important;
        border: 1px solid #0284c7 !important;
    }

    /* Prediction Button Customization */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.39);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5);
    }

    /* Result Box Styling */
    .result-box {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.9) 0%, rgba(4, 47, 46, 0.9) 100%);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);
    }
    .result-text {
        color: #34d399;
        font-size: 24px;
        font-weight: bold;
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

# Input Form
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
            <span style="color: #e2e8f0;">Predicted Outcome / Class:</span>
            <div class="result-text">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.snow()
