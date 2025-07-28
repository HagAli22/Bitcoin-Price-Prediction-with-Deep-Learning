import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from utils.data_preprocessing import DataPreprocessor
from config.config import Config

class GRUPredictor:
    def __init__(self, model_path, scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.window_size = Config.WINDOW_SIZE
        
    def load_model_and_scaler(self):
        """Load the trained model and scaler"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Model and scaler loaded successfully!")
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            
    def prepare_input_data(self, df, target_col='close'):
        """Prepare input data for prediction"""
        # Scale the data using the loaded scaler
        scaled_data = pd.DataFrame(
            self.scaler.transform(df), 
            columns=df.columns, 
            index=df.index
        )
        
        # Get features (exclude target column)
        features = scaled_data.drop(columns=[target_col])
        
        # Take the last window_size samples for prediction
        if len(features) >= self.window_size:
            input_sequence = features.values[-self.window_size:]  # Shape: (window_size, n_features)
            input_sequence = input_sequence.reshape(1, self.window_size, -1)  # Shape: (1, window_size, n_features)
            return input_sequence
        else:
            raise ValueError(f"Need at least {self.window_size} samples for prediction")
    
    def predict_next_price(self, df, target_col='close'):
        """Predict the next price point"""
        if self.model is None or self.scaler is None:
            self.load_model_and_scaler()
            
        # Prepare input data
        input_data = self.prepare_input_data(df, target_col)
        
        # Make prediction
        scaled_prediction = self.model.predict(input_data, verbose=0)
        
        # Inverse transform to get actual price
        # Create a dummy array with the same number of features
        dummy_features = np.zeros((1, df.shape[1]))
        target_col_idx = df.columns.get_loc(target_col)
        dummy_features[0, target_col_idx] = scaled_prediction[0, 0]
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy_features)
        actual_prediction = inverse_transformed[0, target_col_idx]
        
        return actual_prediction
    
    def predict_with_confidence(self, df, n_predictions=10, target_col='close'):
        """Make multiple predictions to estimate confidence interval"""
        if self.model is None or self.scaler is None:
            self.load_model_and_scaler()
            
        predictions = []
        for _ in range(n_predictions):
            pred = self.predict_next_price(df, target_col)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'confidence_interval_95': (mean_pred - 1.96*std_pred, mean_pred + 1.96*std_pred),
            'all_predictions': predictions
        }
    
    def predict_multiple_steps(self, df, n_steps=5, target_col='close'):
        """Predict multiple future price points"""
        if self.model is None or self.scaler is None:
            self.load_model_and_scaler()
            
        predictions = []
        current_df = df.copy()
        
        for step in range(n_steps):
            # Predict next price
            next_price = self.predict_next_price(current_df, target_col)
            predictions.append(next_price)
            
            # Create new row with predicted price
            last_row = current_df.iloc[-1].copy()
            last_row[target_col] = next_price
            
            # Add the new row to the dataframe
            new_index = current_df.index[-1] + pd.Timedelta(days=1)
            current_df.loc[new_index] = last_row
            
        return predictions

def main():
    """Example usage of GRU predictor"""
    # Paths to saved model and scaler
    model_path = 'saved_models/btc_gru_model.h5'
    scaler_path = 'saved_models/gru_scaler.pkl'
    
    # Initialize predictor
    predictor = GRUPredictor(model_path, scaler_path)
    
    # Load some sample data
    try:
        df = pd.read_csv(Config.PROCESSED_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Use the last part of the data for prediction
        recent_data = df.tail(100)
        
        # Make a single prediction
        print("Making single price prediction...")
        next_price = predictor.predict_next_price(recent_data)
        print(f"Predicted next Bitcoin price: ${next_price:.2f}")
        
        # Make prediction with confidence interval
        print("\nMaking prediction with confidence interval...")
        confidence_pred = predictor.predict_with_confidence(recent_data)
        print(f"Mean prediction: ${confidence_pred['mean_prediction']:.2f}")
        print(f"95% Confidence Interval: ${confidence_pred['confidence_interval_95'][0]:.2f} - ${confidence_pred['confidence_interval_95'][1]:.2f}")
        
        # Make multiple predictions
        print("\nMaking multiple price predictions...")
        future_prices = predictor.predict_multiple_steps(recent_data, n_steps=5)
        for i, price in enumerate(future_prices, 1):
            print(f"Day +{i} predicted price: ${price:.2f}")
            
    except FileNotFoundError:
        print("Please ensure you have the processed data and trained model files.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()