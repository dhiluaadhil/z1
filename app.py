import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Health Risk Predictor", page_icon="🏥")
st.title("🏥 Health Risk Score Predictor")

# Load Assets
if os.path.exists('health_risk_model.pkl') and os.path.exists('smoking_le.pkl'):
    model = joblib.load('health_risk_model.pkl')
    le = joblib.load('smoking_le.pkl')

    with st.form("health_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 30)
            bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
            bp = st.number_input("Blood Pressure", 50, 200, 120)
            chol = st.number_input("Cholesterol", 100, 400, 190)
            gluc = st.number_input("Glucose", 50, 300, 100)
            ins = st.number_input("Insulin", 0, 100, 10)
        
        with col2:
            hr = st.number_input("Heart Rate", 40, 200, 70)
            activity = st.slider("Activity Level (1-10)", 1, 10, 5)
            diet = st.slider("Diet Quality (1-10)", 1, 10, 5)
            smoke = st.selectbox("Smoking Status", ["No", "Yes"])
            alc = st.number_input("Alcohol Intake (units/week)", 0, 50, 2)

        if st.form_submit_button("Predict Risk Score"):
            smoke_num = 1 if smoke == "Yes" else 0
            features = np.array([[age, bmi, bp, chol, gluc, ins, hr, activity, diet, smoke_num, alc]])
            prediction = model.predict(features)
            st.success(f"### Estimated Health Risk Score: **{prediction[0]:.2f}**")
else:
    st.error("Missing .pkl files! Please upload 'health_risk_model.pkl' and 'smoking_le.pkl' to GitHub.")