import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_preprocessing import DataPreprocessor

class TestData:
    
    def setup_method(self):
        """Setup test data"""
        # Create sample test dataset
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(50, 15, 1000),
            'feature2': np.random.normal(30, 10, 1000),
            'feature3': np.random.exponential(2, 1000),
            'feature4': np.random.uniform(0, 100, 1000),
            'feature5': np.random.gamma(2, 2, 1000),
            'target': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
        }
        
        # Add some missing values and outliers for testing
        data['feature1'][::100] = np.nan  # 1% missing values
        data['feature2'][50:55] = -999    # Outliers
        
        self.test_df = pd.DataFrame(data)
        self.test_data_path = 'test_preprocessing_data.csv'
        self.test_df.to_csv(self.test_data_path, index=False)
        
        self.preprocessor = DataPreprocessor()
    
    def teardown_method(self):
        """Clean up test files"""
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
    
    def test_load_data(self):
        """Test data loading functionality"""
        df = self.preprocessor.load_data(self.test_data_path)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) == 6  # 5 features + 1 target
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df_with_missing = self.test_df.copy()
        df_cleaned = self.preprocessor.handle_missing_values(df_with_missing)
        
        # Check that missing values are handled
        assert df_cleaned['feature1'].isnull().sum() == 0
        assert len(df_cleaned) > 0
    
    def test_remove_outliers(self):
        """Test outlier removal"""
        df_with_outliers = self.test_df.copy()
        df_no_outliers = self.preprocessor.remove_outliers(df_with_outliers, ['feature2'])
        
        # Should have fewer rows after outlier removal
        assert len(df_no_outliers) <= len(df_with_outliers)
        # No extreme outliers should remain
        assert df_no_outliers['feature2'].min() > -500
    
    def test_normalize_features(self):
        """Test feature normalization"""
        df_clean = self.preprocessor.handle_missing_values(self.test_df.copy())
        df_normalized = self.preprocessor.normalize_features(df_clean)
        
        # Check that features are normalized (approximately mean=0, std=1)
        feature_cols = [col for col in df_normalized.columns if col != 'target']
        for col in feature_cols:
            assert abs(df_normalized[col].mean()) < 0.1
            assert abs(df_normalized[col].std() - 1.0) < 0.1
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        df_processed = self.preprocessor.preprocess(self.test_data_path)
        
        assert isinstance(df_processed, pd.DataFrame)
        assert len(df_processed) > 0
        assert df_processed.isnull().sum().sum() == 0  # No missing values
        
        # Check that target column exists
        assert 'target' in df_processed.columns
        
        # Check feature normalization
        feature_cols = [col for col in df_processed.columns if col != 'target']
        for col in feature_cols:
            assert abs(df_processed[col].mean()) < 0.2
    
    def test_split_features_target(self):
        """Test feature-target splitting"""
        df_processed = self.preprocessor.preprocess(self.test_data_path)
        X, y = self.preprocessor.split_features_target(df_processed)
        
        assert len(X) == len(y)
        assert len(X.columns) == 5  # 5 features
        assert 'target' not in X.columns
        assert all(val in [0, 1] for val in y.unique())