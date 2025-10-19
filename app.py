import streamlit as st
import pandas as pd
import numpy as np
import joblib
import subprocess
import time
from sklearn.preprocessing import StandardScaler

# ✅ Load model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

st.title("💉 Diabetes Predictor (with Auto-Retrain)")
st.write("Predict diabetes using the latest trained model, and retrain automatically with new data!")

# --- User Input Section ---
st.header("Enter Patient Data")

def user_input():
    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 0, 100, 30)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    return pd.DataFrame([data])

input_df = user_input()

# --- Prediction ---
if st.button("Predict"):
    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    st.success(f"Prediction: {'Diabetic 🩸' if pred==1 else 'Not Diabetic ✅'}")
    st.info(f"Probability of Diabetes: {proba*100:.2f}%")

# --- Add new data to dataset ---
import os

# Check if running on Streamlit Cloud (read-only environment)
is_cloud = os.getenv("STREAMLIT_SHARING_MODE") or not os.access(".", os.W_OK)

if not is_cloud:
    st.subheader("Add Data and Retrain")
    st.write("Append this new record to the dataset and retrain the model.")

    outcome = st.selectbox("True Outcome (if known)", [0, 1])
    if st.button("Save Data and Retrain"):
        # Append to CSV
        input_df["Outcome"] = outcome
        df_existing = pd.read_csv("Dataset/diabetes.csv")
        df_updated = pd.concat([df_existing, input_df], ignore_index=True)
        df_updated.to_csv("Dataset/diabetes.csv", index=False)
        st.success("✅ New data saved!")

        # Retrain model using train.py
        with st.spinner("Retraining model... this may take a few minutes ⏳"):
            subprocess.run(["python", "train.py"], check=True)
            time.sleep(2)
        
        st.success("🎉 Model retrained successfully!")
        st.info("Reload the app to use the new model.")
else:
    st.info("ℹ️ This is a demo deployment. Model retraining is only available when running locally.")

