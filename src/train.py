import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib  # Added for saving the scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 1. Load Data
def load_data(path):
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 
        'slope', 'ca', 'thal', 'target'
    ]
    df = pd.read_csv(path, names=columns, na_values='?')
    df = df.dropna()
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

# 2. Feature Engineering & Splitting
def prepare_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SAVE THE SCALER (Crucial for reproducibility)
    joblib.dump(scaler, "scaler.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 3. Train and Log
def train_model(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        
        # Predictions
        predictions = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        auc = roc_auc_score(y_test, probs)
        
        # Logging
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("roc_auc", auc)
        
        # Log Model AND Scaler
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("scaler.pkl")  # Saving the scaler file to MLflow
        
        print(f"{model_name} - Accuracy: {acc:.4f}")

if __name__ == "__main__":
    data_path = "data/heart.csv"
    print("Loading data...")
    df = load_data(data_path)
    
    print("Preparing features and saving scaler...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train Random Forest (Our Best Model)
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    train_model("Random_Forest", rf, X_train, X_test, y_train, y_test)
    
    print("Training complete.")