from prometheus_flask_exporter import PrometheusMetrics
from flask import Flask, request, jsonify
import joblib
import numpy as np
import mlflow.sklearn
import pandas as pd
import os

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Load the model and scaler when the app starts
try:
    # We are loading the files we manually copied into the Docker image
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model/model.pkl") 
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get JSON data
        data = request.get_json()
        
        # Convert JSON to DataFrame (expecting list of features)
        # Example input: {"features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]}
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale the data
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled).max()
        
        result = {
            'prediction': int(prediction[0]), # 0 or 1
            'confidence': float(probability)
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)