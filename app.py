import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("Logistic_Regression_Heart_Model.pkl")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("columns.pkl")

st.title("Heart Disease Predictor")
st.markdown("Please enter your health details : ")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex",['M','F'])
chest_pain = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ['No', 'Yes']
)
resting_ecg = st.selectbox("Resting ECG", ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ['upsloping', 'flat', 'downsloping']) 

if st.button("Predict"):
    raw_input = {
        'RestingBP': resting_bp,
        'Age' : age,
        'FastingBS': 1 if fasting_bs == 'Yes' else 0,
        'Cholesterol': cholesterol,
        'MaxHR': max_hr,
        'Oldpeak' : oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ST_Slope_' + st_slope: 1,
        'ExerciseAngina_' + exercise_angina: 1,
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0 

    input_df = input_df[expected_cols]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0] 

    if prediction == 1:
        st.error("You are at risk of heart disease. Please consult a doctor.")
    else:
        st.success("You are not at risk of heart disease. Keep up the healthy lifestyle!")
