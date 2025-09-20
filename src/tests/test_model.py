import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.train import ModelTrainer
from model.predict import ModelPredictor

class TestModel:
    
    def setup_method(self):
        """Setup test data"""
        # Create sample test data
        np.random.seed(42)
        data = np.random.rand(100, 5)
        target = np.random.randint(0, 2, 100)
        
        df = pd.DataFrame(data, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        df['target'] = target
        
        self.test_data_path = 'test_dataset.csv'
        df.to_csv(self.test_data_path, index=False)
    
    def teardown_method(self):
        """Clean up test files"""
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
        if os.path.exists('models/trained_model.pkl'):
            os.remove('models/trained_model.pkl')
    
    def test_model_training(self):
        """Test model training functionality"""
        trainer = ModelTrainer()
        accuracy = trainer.train_model(self.test_data_path)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert os.path.exists('models/trained_model.pkl')
    
    def test_model_prediction(self):
        """Test model prediction functionality"""
        # Train model first
        trainer = ModelTrainer()
        trainer.train_model(self.test_data_path)
        
        # Test prediction
        predictor = ModelPredictor()
        model = predictor.load_model()
        
        # Test with sample data
        test_features = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])
        prediction = predictor.predict(test_features)
        
        assert prediction is not None
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]