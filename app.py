import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Academic Score Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for Light Sky-Blue Background and Big Sparkling Golden Stars
st.markdown(
    """
    <style>
    /* Light Smooth Sky-Blue Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 50%, #f0f9ff 100%);
        font-family: 'Inter', sans-serif;
        position: relative;
        overflow-x: hidden;
    }

    /* Big Sparkling Golden Stars Background Effect for Light Theme */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            /* Large prominent golden stars */
            radial-gradient(7px 7px at 60px 90px, #d97706, rgba(0,0,0,0)),
            radial-gradient(8px 8px at 220px 280px, #b45309, rgba(0,0,0,0)),
            radial-gradient(6px 6px at 380px 120px, #d97706, rgba(0,0,0,0)),
            radial-gradient(7.5px 7.5px at 540px 350px, #b45309, rgba(0,0,0,0)),
            radial-gradient(8px 8px at 720px 180px, #d97706, rgba(0,0,0,0)),
            radial-gradient(6.5px 6.5px at 880px 410px, #b45309, rgba(0,0,0,0)),
            /* Medium and smaller stars for depth */
            radial-gradient(3.5px 3.5px at 140px 450px, #d97706, rgba(0,0,0,0)),
            radial-gradient(3px 3px at 300px 530px, #f59e0b, rgba(0,0,0,0)),
            radial-gradient(4px 4px at 460px 500px, #d97706, rgba(0,0,0,0)),
            radial-gradient(2.5px 2.5px at 630px 470px, #f59e0b, rgba(0,0,0,0)),
            radial-gradient(4.5px 4.5px at 820px 550px, #b45309, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 950px 650px;
        opacity: 0.85;
        animation: starSparkle 3.5s ease-in-out infinite alternate;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes starSparkle {
        0% { transform: scale(1); opacity: 0.6; filter: drop-shadow(0 0 2px #f59e0b); }
        50% { transform: scale(1.08); opacity: 1; filter: drop-shadow(0 0 6px #d97706); }
        100% { transform: scale(1); opacity: 0.6; filter: drop-shadow(0 0 2px #f59e0b); }
    }

    /* Ensure content stays above background elements */
    .main > div, div[data-testid="stVerticalBlock"] > div.element-container {
        position: relative;
        z-index: 1;
    }

    /* Card Container Style */
    .stCard {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
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
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .result-text {
        color: #065f46;
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
            <span>Predicted Outcome / Class:</span>
            <div class="result-text">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.snow()
