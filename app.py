import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Academic Score Predictor", page_icon="🎓", layout="centered"
)

# Custom CSS for Modern Styling, Glassmorphism, and Animations
st.markdown(
    """
    <style>
    /* Main Background with subtle gradient mesh */
    .stApp {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Title Styling */
    h1 {
        color: #1f2937;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* Card Container Style with Glassmorphism touch */
    .css-1r6slb0, .stContainer {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Prediction Button Customization */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 14px 24px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px 0 rgba(124, 58, 237, 0.6);
        background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
    }

    /* Result Box Styling with Entrance Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-top: 24px;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.2);
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    .result-label {
        font-size: 16px;
        color: #047857;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-text {
        color: #065f46;
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
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
    "Input student subject scores below to generate total predictions using your KNN machine learning model."
)

st.markdown("---")

# Input Form Layout
with st.container():
    st.write("### 📝 Enter Subject Scores (0 - 100)")

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
    st.info(f"📊 **Calculated Total Score:** {calculated_total} / 600")

# Prediction Action
if st.button("🔮 Run Prediction"):
    # Trigger celebratory visual effects
    st.balloons()

    # Format features matching model schema: ['Hindi', 'English', 'Science', 'Maths', 'History', 'Geography', 'Total']
    features = np.array(
        [[hindi, english, science, maths, history, geography, calculated_total]]
    )

    # Perform Prediction
    prediction = model.predict(features)[0]

    # Display Pretty Result Output with Custom CSS card
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Predicted Outcome / Class</div>
            <div class="result-text">{prediction}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
