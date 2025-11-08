"""
MLflow Experiment Tracking for Diabetes Prediction Models
This script replicates the model training pipeline with MLflow integration
to track experiments, parameters, metrics, and artifacts.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set MLflow tracking URI (you can change this to a remote server if needed)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Diabetes_Prediction_Models")

def load_and_preprocess_data(data_path):
    """Load and preprocess the diabetes dataset"""
    print("Loading and preprocessing data...")
    data = pd.read_csv(data_path)
    
    # Feature Selection (dropping SkinThickness and BloodPressure as per analysis)
    X = data.drop(columns=['Outcome', 'SkinThickness', 'BloodPressure'], axis=1)
    y = data['Outcome']
    
    return X, y, data

def handle_invalid_zeros_and_split(X, y):
    """Handle zero values and split the data"""
    print("Handling invalid zero values...")
    invalid_zero = ['Glucose', 'Insulin', 'BMI']
    
    # Replace zeros with NaN
    for col in invalid_zero:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
    
    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate medians from training set
    medians = {}
    for col in invalid_zero:
        medians[col] = X_train[col].median()
    
    # Fill NaNs with training medians
    for col in invalid_zero:
        X_train[col].fillna(medians[col], inplace=True)
        X_test[col].fillna(medians[col], inplace=True)
    
    return X_train, X_test, y_train, y_test, medians, invalid_zero

def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    print("Applying SMOTE for class balancing...")
    sm = SMOTE(random_state=42, sampling_strategy='minority')
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model(model, X_test, y_test, y_pred):
    """Calculate all evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm

def plot_confusion_matrix(cm, model_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    # Save figure
    filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def train_model_with_mlflow(model_name, model, X_train, y_train, X_test, y_test, feature_columns):
    """Train a model and log everything to MLflow"""
    
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("features", ', '.join(feature_columns))
        
        # Log model-specific parameters
        if hasattr(model, 'get_params'):
            for param, value in model.get_params().items():
                mlflow.log_param(f"model_{param}", value)
        
        # Perform cross-validation
        print(f"Performing 5-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        mlflow.log_metric("cv_recall_mean", cv_mean)
        mlflow.log_metric("cv_recall_std", cv_std)
        print(f"Cross-Validation Recall: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        # Train the model
        print(f"Training final model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        metrics, cm = evaluate_model(model, X_test, y_test, y_pred)
        
        # Log metrics
        print(f"\nTest Set Metrics:")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Plot and log confusion matrix
        cm_filename = plot_confusion_matrix(cm, model_name)
        mlflow.log_artifact(cm_filename)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\n✓ {model_name} logged successfully to MLflow")
        
        return model, metrics, cv_mean

def train_best_model_with_grid_search(X_train, y_train, X_test, y_test, feature_columns):
    """Train Random Forest with GridSearchCV and log to MLflow"""
    
    with mlflow.start_run(run_name="Random Forest (GridSearchCV)"):
        print(f"\n{'='*60}")
        print(f"Training Random Forest with GridSearchCV...")
        print(f"{'='*60}")
        
        # Log basic parameters
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("hyperparameter_tuning", "GridSearchCV")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("features", ', '.join(feature_columns))
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Log parameter grid
        mlflow.log_param("param_grid", str(param_grid))
        
        # Initialize Random Forest and GridSearchCV
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf, 
            param_grid=param_grid,
            cv=5, 
            scoring='recall',
            n_jobs=-1,
            verbose=1
        )
        
        print("Performing GridSearchCV (this may take a while)...")
        grid_search.fit(X_train, y_train)
        
        # Log best parameters
        print(f"\nBest Parameters: {grid_search.best_params_}")
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Evaluate model
        metrics, cm = evaluate_model(best_model, X_test, y_test, y_pred)
        
        # Log metrics
        print(f"\nTest Set Metrics:")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Plot and log confusion matrix
        cm_filename = plot_confusion_matrix(cm, "Random Forest GridSearch")
        mlflow.log_artifact(cm_filename)
        
        # Log the model
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\n✓ Random Forest (GridSearchCV) logged successfully to MLflow")
        
        return best_model, metrics, grid_search.best_params_

def save_best_artifacts(model, scaler, medians, feature_columns):
    """Save the best model artifacts"""
    artifacts = {
        "model": model,
        "scaler": scaler,
        "medians": medians,
        "feature_order": feature_columns
    }
    
    joblib.dump(artifacts, "diabetes_artifacts.pkl")
    print("\n✓ Saved best model artifacts to diabetes_artifacts.pkl")

def main():
    """Main execution function"""
    print("="*60)
    print("DIABETES PREDICTION - MLflow EXPERIMENT TRACKING")
    print("="*60)
    
    # Load and preprocess data
    data_path = "Dataset/diabetes.csv"
    X, y, data = load_and_preprocess_data(data_path)
    feature_columns = list(X.columns)
    
    # Handle invalid zeros and split data
    X_train, X_test, y_train, y_test, medians, invalid_zero = handle_invalid_zeros_and_split(X, y)
    
    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_resampled, X_test)
    
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "SVM": SVC(class_weight="balanced", random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    }
    
    # Train all models with MLflow tracking
    results = {}
    for model_name, model in models.items():
        trained_model, metrics, cv_score = train_model_with_mlflow(
            model_name, model, X_train_scaled, y_train_resampled, 
            X_test_scaled, y_test, feature_columns
        )
        results[model_name] = {
            'model': trained_model,
            'metrics': metrics,
            'cv_score': cv_score
        }
    
    # Train Random Forest with GridSearchCV
    best_model, best_metrics, best_params = train_best_model_with_grid_search(
        X_train_scaled, y_train_resampled, X_test_scaled, y_test, feature_columns
    )
    
    # Save the best model artifacts
    save_best_artifacts(best_model, scaler, medians, feature_columns)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print("\nModel Comparison (CV Recall Score):")
    for model_name, result in results.items():
        print(f"  {model_name}: {result['cv_score']:.4f}")
    
    print("\n\nBest Model: Random Forest (GridSearchCV)")
    print(f"Best Parameters: {best_params}")
    print("\nTest Set Performance:")
    for metric_name, metric_value in best_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n" + "="*60)
    print("All experiments logged to MLflow!")
    print("Run 'mlflow ui' to view the dashboard")
    print("="*60)

if __name__ == "__main__":
    main()
