import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for ML pipeline
    Handles missing values, outliers, normalization, and feature engineering
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.target_column = 'target'
    
    def load_data(self, file_path):
        """
        Load data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File {file_path} not found!")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self, df):
        """
        Basic data exploration and summary statistics
        
        Args:
            df (pandas.DataFrame): Input dataset
            
        Returns:
            dict: Data summary statistics
        """
        summary = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'summary_stats': df.describe().to_dict()
        }
        
        logger.info(f"Dataset shape: {summary['shape']}")
        logger.info(f"Missing values: {sum(summary['missing_values'].values())}")
        
        return summary
    
    def handle_missing_values(self, df, strategy='median'):
        """
        Handle missing values in the dataset
        
        Args:
            df (pandas.DataFrame): Input dataset
            strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
            
        Returns:
            pandas.DataFrame: Dataset with missing values handled
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
            logger.info("Missing values dropped")
        else:
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != self.target_column]
            
            if len(numeric_columns) > 0:
                if strategy == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif strategy == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                else:
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                
                df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
                logger.info(f"Missing values imputed using {strategy} strategy")
        
        return df_clean
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=3):
        """
        Remove outliers from specified columns
        
        Args:
            df (pandas.DataFrame): Input dataset
            columns (list): Columns to check for outliers
            method (str): Method to use ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pandas.DataFrame: Dataset with outliers removed
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col != self.target_column]
        
        initial_len = len(df_clean)
        
        for column in columns:
            if method == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean = df_clean[
                    (df_clean[column] >= lower_bound) & 
                    (df_clean[column] <= upper_bound)
                ]
            
            elif method == 'zscore':
                z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
                df_clean = df_clean[z_scores < threshold]
        
        removed_count = initial_len - len(df_clean)
        logger.info(f"Removed {removed_count} outliers using {method} method")
        
        return df_clean
    
    def normalize_features(self, df, method='standard'):
        """
        Normalize/scale feature columns
        
        Args:
            df (pandas.DataFrame): Input dataset
            method (str): Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            pandas.DataFrame: Dataset with normalized features
        """
        df_scaled = df.copy()
        
        # Get feature columns (exclude target)
        feature_columns = [col for col in df_scaled.columns if col != self.target_column]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:  # minmax
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        df_scaled[feature_columns] = scaler.fit_transform(df_scaled[feature_columns])
        logger.info(f"Features normalized using {method} scaling")
        
        self.scaler = scaler  # Store for later use
        self.feature_columns = feature_columns
        
        return df_scaled
    
    def create_features(self, df):
        """
        Create additional features (feature engineering)
        
        Args:
            df (pandas.DataFrame): Input dataset
            
        Returns:
            pandas.DataFrame: Dataset with additional features
        """
        df_features = df.copy()
        
        # Get numeric columns for feature creation
        numeric_cols = [col for col in df_features.select_dtypes(include=[np.number]).columns 
                       if col != self.target_column]
        
        if len(numeric_cols) >= 2:
            # Create interaction features
            df_features[f'{numeric_cols[0]}_x_{numeric_cols[1]}'] = (
                df_features[numeric_cols[0]] * df_features[numeric_cols[1]]
            )
            
            # Create ratio features
            df_features[f'{numeric_cols[0]}_div_{numeric_cols[1]}'] = (
                df_features[numeric_cols[0]] / (df_features[numeric_cols[1]] + 1e-8)
            )
            
            # Create sum features
            df_features['feature_sum'] = df_features[numeric_cols].sum(axis=1)
            
            logger.info("Additional features created")
        
        return df_features
    
    def split_features_target(self, df):
        """
        Split dataset into features and target
        
        Args:
            df (pandas.DataFrame): Input dataset
            
        Returns:
            tuple: (X, y) features and target
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        logger.info(f"Data split into features ({X.shape}) and target ({y.shape})")
        
        return X, y
    
    def preprocess(self, file_path, remove_outliers=True, create_additional_features=False):
        """
        Complete preprocessing pipeline
        
        Args:
            file_path (str): Path to the dataset
            remove_outliers (bool): Whether to remove outliers
            create_additional_features (bool): Whether to create additional features
            
        Returns:
            pandas.DataFrame: Fully preprocessed dataset
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        
        # Explore data
        self.explore_data(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy='median')
        
        # Remove outliers if requested
        if remove_outliers:
            df = self.remove_outliers(df, method='iqr')
        
        # Create additional features if requested
        if create_additional_features:
            df = self.create_features(df)
        
        # Normalize features
        df = self.normalize_features(df, method='standard')
        
        logger.info("Preprocessing pipeline completed successfully!")
        
        return df

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Process the dataset
    try:
        processed_data = preprocessor.preprocess('dataset.csv')
        print("Preprocessing completed successfully!")
        print(f"Final dataset shape: {processed_data.shape}")
        
        # Save processed data
        processed_data.to_csv('processed_dataset.csv', index=False)
        logger.info("Processed dataset saved to 'processed_dataset.csv'")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")