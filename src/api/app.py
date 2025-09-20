from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from src.model.predict import ModelPredictor

app = Flask(__name__)

# Load model on startup
try:
    predictor = ModelPredictor()
    model = predictor.load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid input format"}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        prediction = predictor.predict(features)
        
        return jsonify({
            "prediction": prediction.tolist(),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)