# Diabetes Prediction with MLflow Tracking

This project implements a diabetes prediction system with MLflow experiment tracking and a Streamlit web interface.

## 📁 Project Structure

```
PIMA/
├── Dataset/
│   └── diabetes.csv              # Dataset
├── PIMA_Diabetes.ipynb           # Original notebook
├── train_with_mlflow.py          # MLflow training script
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── diabetes_artifacts.pkl        # Saved model artifacts
└── mlruns/                       # MLflow tracking data (auto-generated)
```

## 🚀 Getting Started

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Train Models with MLflow Tracking

Run the training script to train all models and track experiments:

```powershell
python train_with_mlflow.py
```

This script will:
- Train **Logistic Regression**, **SVM**, and **Random Forest** models
- Perform **5-fold cross-validation** for each model
- Apply **GridSearchCV** for Random Forest hyperparameter tuning
- Log all **parameters**, **metrics**, and **artifacts** to MLflow
- Save the best model as `diabetes_artifacts.pkl`

### 3. View MLflow Dashboard

After training, launch the MLflow UI to compare experiments:

```powershell
mlflow ui
```

Then open your browser to: `http://localhost:5000`

### 4. Run Streamlit Application

Launch the web interface for predictions:

```powershell
streamlit run app.py
```

The app will open in your browser at: `http://localhost:8501`

## 📊 What MLflow Tracks

### Parameters
- Model type and hyperparameters
- Train-test split ratio
- SMOTE settings
- Feature names
- GridSearchCV parameter grid

### Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Cross-validation scores (mean and std)

### Artifacts
- Trained model (sklearn format)
- Confusion matrix plots
- Feature importance (for tree-based models)

## 🎯 Models Trained

1. **Logistic Regression**
   - Max iterations: 1000
   - Class weight: balanced

2. **Support Vector Machine (SVM)**
   - Kernel: RBF (default)
   - Class weight: balanced

3. **Random Forest** (baseline)
   - N estimators: 100
   - Class weight: balanced

4. **Random Forest (GridSearchCV)** ⭐ Best Model
   - Hyperparameter tuning with 5-fold CV
   - Optimized for recall score

## 🔍 Features Used

- Pregnancies
- Glucose
- Insulin
- BMI (Body Mass Index)
- DiabetesPedigreeFunction
- Age

**Note:** BloodPressure and SkinThickness were excluded due to low correlation with the outcome.

## 📈 Data Preprocessing

1. **Invalid Zero Handling**: Glucose, Insulin, and BMI zeros replaced with median values
2. **SMOTE**: Applied to handle class imbalance
3. **Scaling**: StandardScaler for feature normalization
4. **Train-Test Split**: 80-20 split with stratification

## 🎨 Streamlit App Features

- Interactive input form for patient data
- Real-time predictions with probability scores
- Risk level visualization
- Input validation and helpful tooltips
- Model information sidebar

## 📝 Notes

- The model is trained on the PIMA Indians Diabetes Database
- All preprocessing steps match the original notebook analysis
- MLflow experiments are stored locally in the `mlruns/` directory
- The best model is automatically saved as `diabetes_artifacts.pkl`

## 🛠️ Troubleshooting

**Issue**: MLflow UI not loading
- **Solution**: Check if port 5000 is available, or specify a different port: `mlflow ui --port 5001`

**Issue**: Streamlit app can't find model file
- **Solution**: Ensure `diabetes_artifacts.pkl` exists in the same directory. Run `train_with_mlflow.py` first.

**Issue**: Module not found errors
- **Solution**: Reinstall dependencies: `pip install -r requirements.txt`

## 📧 Contact

For questions or issues, please refer to the GitHub repository.
