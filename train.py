# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import mlflow
import mlflow.sklearn
import os

# ✅ Load and clean dataset
# Use relative path instead of hardcoded absolute path
dataset_path = os.path.join("Dataset", "diabetes.csv")
data = pd.read_csv(dataset_path)

# Replace invalid zeros with NaN for these columns
cols_with_zero_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols_with_zero_nan] = data[cols_with_zero_nan].replace(0, np.nan)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Define models and hyperparameter grids
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["liblinear", "lbfgs"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10]
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf", "poly"],
            "model__gamma": ["scale", "auto"]
        }
    }
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Diabetes_Predictor")

best_model = None
best_score = 0
best_name = ""

for name, cfg in models.items():
    print(f"\n🔍 Training {name} with cross-validation...")

    # Full pipeline = preprocessing + model
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
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

        # Log params and metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")

        # Keep best model
        if auc > best_score:
            best_model = grid.best_estimator_
            best_score = auc
            best_name = name

# ✅ Save best model for Streamlit app
joblib.dump(best_model, "model.joblib")
print(f"\n🏆 Best model: {best_name} with AUC = {best_score:.3f}")
print("✅ Model saved as model.joblib")
