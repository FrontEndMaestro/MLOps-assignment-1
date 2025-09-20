import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_preprocessing import DataPreprocessor
from model.train import ModelTrainer
from model.predict import ModelPredictor

class TestMLPipeline:
    
    def setup_method(self):
        """Setup test environment"""
        np.random.seed(42)
        # Create minimal test dataset
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100), 
            'feature3': np.random.normal(0, 1, 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data.to_csv(self.test_file.name, index=False)
        self.test_file.close()
        
    def teardown_method(self):
        """Cleanup"""
        if os.path.exists(self.test_file.name):
            os.remove(self.test_file.name)
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline"""
        preprocessor = DataPreprocessor()
        
        # Test data loading
        df = preprocessor.load_data(self.test_file.name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Test preprocessing pipeline
        processed_df = preprocessor.preprocess(self.test_file.name)
        assert isinstance(processed_df, pd.DataFrame)
        assert processed_df.isnull().sum().sum() == 0
    
    def test_model_training(self):
        """Test model training functionality"""
        # Use the actual dataset
        trainer = ModelTrainer()
        accuracy = trainer.train_model('src/data/dataset.csv')
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
        assert os.path.exists('models/trained_model.pkl')
        assert os.path.exists('models/scaler.pkl')
    
    def test_model_prediction(self):
        """Test model prediction functionality"""
        predictor = ModelPredictor()
        model = predictor.load_model()
        
        # Test with sample data
        sample_features = np.array([[45, 5000, 75, 2500, 3, 12, 60, 6.5, 24, 4]])
        result = predictor.predict(sample_features)
        
        assert 'prediction' in result
        assert 'probability' in result
        assert result['prediction'][0] in [0, 1]
        assert len(result['probability']) == 1
        assert len(result['probability'][0]) == 2  # Two classes
