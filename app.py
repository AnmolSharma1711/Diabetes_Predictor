import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Load the saved artifacts
@st.cache_resource
def load_artifacts():
    try:
        artifacts = joblib.load("diabetes_artifacts.pkl")
        return artifacts
    except FileNotFoundError:
        st.error("Model file 'diabetes_artifacts.pkl' not found. Please ensure the file exists in the same directory.")
        return None

# Main app
def main():
    st.title("🏥 Diabetes Prediction System")
    st.markdown("""
    This application predicts the likelihood of diabetes based on various health parameters.
    Please enter the patient's information below.
    """)
    
    # Load artifacts
    artifacts = load_artifacts()
    
    if artifacts is None:
        st.stop()
    
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    medians = artifacts["medians"]
    feature_order = artifacts["feature_order"]
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of times pregnant"
        )
        
        glucose = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=0,
            max_value=300,
            value=120,
            step=1,
            help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)"
        )
        
        insulin = st.number_input(
            "Insulin Level (mu U/ml)",
            min_value=0,
            max_value=900,
            value=0,
            step=1,
            help="2-Hour serum insulin (mu U/ml). Enter 0 if unknown."
        )
    
    with col2:
        st.subheader("Physical Measurements")
        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=0.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Weight in kg/(height in m)^2"
        )
        
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="Diabetes pedigree function (genetic influence)"
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30,
            step=1,
            help="Age in years"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
        # Create input dictionary
        manual_input_dict = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": diabetes_pedigree,
            "Age": age
        }
        
        # Build DataFrame in correct column order
        manual_df = pd.DataFrame(
            [[manual_input_dict[col] for col in feature_order]],
            columns=feature_order
        )
        
        # Define invalid zero columns (same as training)
        invalid_zero = ['Glucose', 'Insulin', 'BMI']
        
        # Zero -> NaN replacement for invalid zero columns
        for col in invalid_zero:
            if col in manual_df.columns:
                manual_df[col] = manual_df[col].replace(0, np.nan)
        
        # Fill NaNs with training medians
        for col in invalid_zero:
            if col in manual_df.columns:
                manual_df[col].fillna(medians[col], inplace=True)
        
        # Scale using stored scaler
        manual_scaled = scaler.transform(manual_df)
        
        # Predict
        prediction = model.predict(manual_scaled)[0]
        prediction_proba = model.predict_proba(manual_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Create three columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            if prediction == 1:
                st.error("⚠️ **High Risk of Diabetes**")
            else:
                st.success("✅ **Low Risk of Diabetes**")
        
        with res_col2:
            st.metric(
                "Probability of No Diabetes",
                f"{prediction_proba[0]:.2%}"
            )
        
        with res_col3:
            st.metric(
                "Probability of Diabetes",
                f"{prediction_proba[1]:.2%}"
            )
        
        # Additional information
        st.markdown("---")
        st.info("""
        **Note:** This prediction is based on machine learning analysis and should not replace professional medical diagnosis. 
        Please consult with a healthcare provider for proper medical evaluation.
        """)
        
        # Show processed input data
        with st.expander("📋 View Processed Input Data"):
            st.dataframe(manual_df, use_container_width=True)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a **Random Forest Classifier** trained on the PIMA Indians Diabetes Database.
        
        **Features Used:**
        - Number of Pregnancies
        - Glucose Level
        - Insulin Level
        - BMI (Body Mass Index)
        - Diabetes Pedigree Function
        - Age
        
        **Note:** Blood Pressure and Skin Thickness were excluded during feature selection as they showed low correlation with the outcome.
        """)
        
        st.markdown("---")
        st.header("📈 Model Performance")
        st.markdown("""
        The model was trained using:
        - **SMOTE** for handling class imbalance
        - **StandardScaler** for feature scaling
        - **GridSearchCV** for hyperparameter tuning
        - **5-fold Cross-Validation** for model selection
        """)
        
        st.markdown("---")
        st.markdown("**Developed using Streamlit**")
        st.markdown("Model: Random Forest Classifier")

if __name__ == "__main__":
    main()
