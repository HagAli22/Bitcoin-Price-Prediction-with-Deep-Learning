import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessor:
    def __init__(self, window_size=30, test_size=0.2, val_size=0.2):
        self.window_size = window_size
        self.test_size = test_size
        self.val_size = val_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_clean_data(self, file_path):
        """Load and clean the dataset"""
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
        df['date'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Sort by date and set as index
        df.sort_values('date', inplace=True)
        df.set_index('date', inplace=True)
        
        # Drop unnecessary columns
        if 'timestamp' in df.columns:
            df.drop(['timestamp'], axis=1, inplace=True)
        if 'index' in df.columns:
            df.drop(['index'], axis=1, inplace=True)
            
        return df
    
    def scale_data(self, df, target_col='close'):
        """Scale the data using MinMaxScaler"""
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(df), 
            columns=df.columns, 
            index=df.index
        )
        return scaled_data
    
    def create_sequences(self, data, target, window_size):
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, target_col='close'):
        """Complete data preparation pipeline"""
        # Scale data
        scaled_data = self.scale_data(df, target_col)
        
        # Prepare features and target
        X_data = scaled_data.drop(columns=[target_col])
        y_data = scaled_data[target_col]
        
        # Create sequences
        X, y = self.create_sequences(X_data.values, y_data.values, self.window_size)
        
        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.val_size, shuffle=False
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'input_shape': (X_train.shape[1], X_train.shape[2])
        }
    
    def save_scaler(self, filepath):
        """Save the fitted scaler"""
        joblib.dump(self.scaler, filepath)
    
    def load_scaler(self, filepath):
        """Load a saved scaler"""
        self.scaler = joblib.load(filepath)