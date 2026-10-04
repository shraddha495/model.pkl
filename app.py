import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Academic Score Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for Dark Sky-Blue Background and Modern Styling
st.markdown(
    """
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #020617 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Container Style */
    .stCard {
        background-color: #1e293b;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }

    /* Text Color Adjustments for Dark Background Readability */
    h1, h2, h3, h4, h5, h6, label, .stMarkdown p {
        color: #f8fafc !important;
    }

    /* Input Fields Styling */
    .stNumberInput input {
        background-color: #334155 !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }

    /* Info Box Customization */
    .stInfo {
        background-color: rgba(30, 41, 59, 0.9) !important;
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
