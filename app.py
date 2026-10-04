import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Academic Score Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for Sky Blue Dark Background, Sparkling Golden Stars, and Modern Styling
st.markdown(
    """
    <style>
    /* Main Background: Dark Sky Blue Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 50%, #0c4a6e 100%);
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
        position: relative;
        overflow-x: hidden;
    }

    /* Sparkling Golden Stars Effect in Background */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #fde047, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 40px 70px, #facc15, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 90px 40px, #fef08a, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 160px 120px, #fde047, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 230px 200px, #facc15, rgba(0,0,0,0)),
            radial-gradient(1.5px 1.5px at 300px 80px, #fef08a, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 380px 280px, #fde047, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 450px 150px, #facc15, rgba(0,0,0,0)),
            radial-gradient(1px 1px at 520px 220px, #fef08a, rgba(0,0,0,0)),
            radial-gradient(2px 2px at 600px 50px, #fde047, rgba(0,0,0,0));
        background-repeat: repeat;
        background-size: 650px 350px;
        opacity: 0.7;
        animation: sparkle 4s linear infinite;
        z-index: 0;
        pointer-events: none;
    }

    @keyframes sparkle {
        0% { transform: translateY(0); opacity: 0.5; }
        50% { opacity: 0.9; }
        100% { transform: translateY(-50px); opacity: 0.5; }
    }

    /* Ensure Streamlit Content Stays Above the Background Stars */
    .stApp > div {
        position: relative;
        z-index: 1;
    }

    /* Card & Container Polish */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        color: #f8fafc;
    }

    /* Input Field Labels & Text */
    label {
        color: #e2e8f0 !important;
        font-weight: 500;
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

    /* Info Box Styling */
    div.stInfo {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid #38bdf8 !important;
        color: #e0f2fe !important;
        border-radius: 8px;
    }

    /* Result Box Styling */
    .result-box {
        background: linear-gradient(135deg, rgba(6, 95, 70, 0.85) 0%, rgba(4, 120, 87, 0.85) 100%);
        border: 1px solid #34d399;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    .result-text {
        color: #a7f3d0;
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
            <span style="color: #ecfdf5;">Predicted Outcome / Class:</span>
            <div class="result-text">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.snow()
