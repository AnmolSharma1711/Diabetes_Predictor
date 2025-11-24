import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except:
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load("scaler.pkl")
        return scaler
    except:
        return None

# Load dataset for visualization
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Dataset/diabetes.csv")
        return data
    except:
        return None

def preprocess_input(data_dict, scaler):
    """Preprocess user input to match training data format"""
    # Ensure column order matches training data (same as in notebook)
    # Order: Pregnancies, Glucose, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    
    # Create array with values in correct order
    input_array = np.array([[
        data_dict['Pregnancies'],
        data_dict['Glucose'],
        data_dict['SkinThickness'],
        data_dict['Insulin'],
        data_dict['BMI'],
        data_dict['DiabetesPedigreeFunction'],
        data_dict['Age']
    ]])
    
    # Apply the same scaler used during training
    scaled_data = scaler.transform(input_array)
    
    return scaled_data

def main():
    # Header
    st.title("🏥 Diabetes Prediction System")
    st.markdown("---")
    
    # Load model and scaler
    model = load_model()
    scaler = load_scaler()
    
    if model is None or scaler is None:
        st.error("⚠️ Model or scaler not found!")
        st.info("Please run the Jupyter notebook cells to train and save the model and scaler first.")
        return
    
    show_prediction_page(model, scaler)

def show_prediction_page(model, scaler):
    st.subheader("Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            help="Number of times pregnant"
        )
        
        glucose = st.slider(
            "Glucose Level (mg/dL)",
            min_value=0,
            max_value=200,
            value=120,
            help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test"
        )
        
        skin_thickness = st.slider(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            help="Triceps skin fold thickness"
        )
        
        insulin = st.slider(
            "Insulin Level (μU/mL)",
            min_value=0,
            max_value=900,
            value=80,
            help="2-Hour serum insulin"
        )
    
    with col2:
        bmi = st.slider(
            "BMI (Body Mass Index)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body mass index (weight in kg/(height in m)^2)"
        )
        
        dpf = st.slider(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=0.5,
            step=0.01,
            help="Diabetes pedigree function (genetic influence)"
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=25,
            help="Age in years"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔍 Predict Diabetes Risk"):
        # Prepare input data (excluding BloodPressure as per the model)
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        # Preprocess and predict
        processed_data = preprocess_input(input_data, scaler)
        prediction = model.predict(processed_data)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error("### ⚠️ High Risk of Diabetes")
            st.warning("The model predicts that this patient has a high risk of diabetes.")
        else:
            st.success("### ✅ Low Risk of Diabetes")
            st.info("The model predicts that this patient has a low risk of diabetes.")
        
        # Risk factors analysis
        st.markdown("---")
        st.subheader("🎯 Risk Factor Analysis")
        
        risk_factors = []
        if glucose > 140:
            risk_factors.append("⚠️ High glucose level (>140 mg/dL)")
        if insulin > 200:
            risk_factors.append("⚠️ High insulin level (>200 μU/mL)")
        if bmi > 35:
            risk_factors.append("⚠️ High BMI (>35)")
        if age > 40:
            risk_factors.append("⚠️ Age over 40 years")
        if dpf > 0.5:
            risk_factors.append("⚠️ High genetic predisposition")
        
        if risk_factors:
            st.warning("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.info("✅ No major risk factors identified based on input values.")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        if prediction[0] == 1:
            st.write("- Consult with a healthcare professional immediately")
            st.write("- Follow a balanced, low-sugar diet")
            st.write("- Engage in regular physical activity")
            st.write("- Monitor blood glucose levels regularly")
        else:
            st.write("- Continue maintaining a healthy diet")
            st.write("- Stay physically active")
            st.write("- Get regular health check-ups")


if __name__ == "__main__":
    main()
