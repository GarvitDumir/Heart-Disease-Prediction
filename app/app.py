import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# LOAD MODEL, SCALER AND COLUMNS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "model" / "knn_heart_disease_model.pkl"
)

scaler = joblib.load(
    BASE_DIR / "model" / "scaler_heart_disease_model.pkl"
)

expected_columns = joblib.load(
    BASE_DIR / "model" / "cols_heart_disease_model.pkl"
)


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient information below")


age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dl)",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.1
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict"):

    # --------------------------------------
    # Create dataframe
    # --------------------------------------

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })


    # --------------------------------------
    # One-hot encode categorical columns
    # --------------------------------------

    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # --------------------------------------
    # Make columns EXACTLY same as training
    # --------------------------------------

    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )


    # --------------------------------------
    # Convert boolean columns to integers
    # --------------------------------------

    input_data = input_data.astype(float)


    # --------------------------------------
    # Scale data
    # --------------------------------------

    input_scaled = scaler.transform(input_data)


    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(input_scaled)[0]


    # --------------------------------------
    # Result
    # --------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Heart Disease Detected"
        )

    else:

        st.success(
            "✅ No Heart Disease Detected"
        )

