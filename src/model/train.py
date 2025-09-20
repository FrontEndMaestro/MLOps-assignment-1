import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def load_data(self, data_path):
        """Load and preprocess data"""
        df = pd.read_csv(data_path)
        # Add your specific preprocessing here
        return df
    
    def train_model(self, data_path):
        """Train the ML model"""
        df = self.load_data(data_path)
        
        # Assuming last column is target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/trained_model.pkl')
        
        return accuracy

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model('src/data/dataset.csv')