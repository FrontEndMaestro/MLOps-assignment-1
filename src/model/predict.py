import joblib
import numpy as np
import os

class ModelPredictor:
    def __init__(self):
        self.model = None
        self.model_path = 'models/trained_model.pkl'
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return self.model
        else:
            raise FileNotFoundError("Trained model not found")
    
    def predict(self, features):
        """Make predictions"""
        if self.model is None:
            self.load_model()
        
        return self.model.predict(features)