# train.py (Updated for SMOTE + class weights)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib #Model Saving
import mlflow #MLOps Logging
import mlflow.sklearn
import os
import sys
from imblearn.over_sampling import SMOTE #Synthetic Minority OverSampling Technique
from imblearn.pipeline import Pipeline as ImbPipeline  # use imblearn's pipeline for SMOTE

# Load dataset ( Used Because Dataset is in Local Disk "D" and python is creating issue with Escape Sequence)
dataset_path = os.path.join("Dataset", "diabetes.csv")
data = pd.read_csv(dataset_path)

#Problem :- 0 values in features like Glucose, BP etc are not possible.
#Solution :- Replace 0 with NaN for these features for Simple Imputation

# Replace zeros with NaN for certain columns
cols_with_zero_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols_with_zero_nan] = data[cols_with_zero_nan].replace(0, np.nan)

#Feature and Target Selection
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cross-validation setup
#Staratified is used to maintain class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# MLflow setup for Logs Tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Diabetes_Predictor")

best_model = None
best_score = 0
best_name = ""

# Define models and hyperparameter grids with class_weight
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42, class_weight="balanced"),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf", "poly"],
            "model__gamma": ["scale", "auto"]
        }
    }
}

# Preprocessing + SMOTE integrated into pipeline
for name, cfg in models.items():
    print(f"\n Training {name} with cross-validation and SMOTE...")

    pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", cfg["model"])
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=cfg["params"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    with mlflow.start_run(run_name=name):
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"{name} → AUC: {auc:.3f}, ACC: {acc:.3f}, F1: {f1:.3f}")

        # Log parameters & metrics to MLflow
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")

        if auc > best_score:
            best_model = grid.best_estimator_
            best_score = auc
            best_name = name

# Save best model
joblib.dump(best_model, "model.joblib")
print(f"\n Best model: {best_name} with AUC = {best_score:.3f}")
print("Model saved as model.joblib")
