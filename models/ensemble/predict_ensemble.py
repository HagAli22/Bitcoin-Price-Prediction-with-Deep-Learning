import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import joblib
from utils.data_preprocessing import DataPreprocessor
from config.config import Config

class EnsemblePredictor:
    def __init__(self, model_path, scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.window_size = Config.WINDOW_SIZE
        
    def load_model_and_scaler(self):
        """Load the trained ensemble model and scaler"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Ensemble model and scaler loaded successfully!")
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
        """Predict the next price point using ensemble"""
        if self.model is None or self.scaler is None:
            self.load_model_and_scaler()
            
        # Prepare input data
        input_data = self.prepare_input_data(df, target_col)
        
        # Make prediction
        scaled_prediction = self.model.predict(input_data)
        
        # Inverse transform to get actual price
        # Create a dummy array with the same number of features
        dummy_features = np.zeros((1, df.shape[1]))
        target_col_idx = df.columns.get_loc(target_col)
        dummy_features[0, target_col_idx] = scaled_prediction[0]
        
        # Inverse transform
        inverse_transformed = self.scaler.inverse_transform(dummy_features)
        actual_prediction = inverse_transformed[0, target_col_idx]
        
        return actual_prediction
    
    def get_individual_predictions(self, df, target_col='close'):
        """Get predictions from individual models in the ensemble"""
        if self.model is None or self.scaler is None:
            self.load_model_and_scaler()
            
        # Prepare input data
        input_data = self.prepare_input_data(df, target_col)
        
        # Get individual model predictions
        individual_preds = {}
        for name, estimator in self.model.named_estimators_.items():
            scaled_pred = estimator.predict(input_data)
            
            # Inverse transform
            dummy_features = np.zeros((1, df.shape[1]))
            target_col_idx = df.columns.get_loc(target_col)
            dummy_features[0, target_col_idx] = scaled_pred[0]
            
            inverse_transformed = self.scaler.inverse_transform(dummy_features)
            actual_pred = inverse_transformed[0, target_col_idx]
            
            individual_preds[name] = actual_pred
        
        # Get ensemble prediction
        ensemble_pred = self.predict_next_price(df, target_col)
        individual_preds['ensemble'] = ensemble_pred
        
        return individual_preds
    
    def predict_with_uncertainty(self, df, target_col='close'):
        """Predict with uncertainty estimation using individual model disagreement"""
        if self.model is None or self.scaler is None:
            self.load_model_and_scaler()
            
        # Get individual predictions
        individual_preds = self.get_individual_predictions(df, target_col)
        
        # Calculate statistics
        model_predictions = [individual_preds[name] for name in self.model.named_estimators_.keys()]
        
        mean_pred = np.mean(model_predictions)
        std_pred = np.std(model_predictions)
        ensemble_pred = individual_preds['ensemble']
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': individual_preds,
            'mean_individual': mean_pred,
            'std_individual': std_pred,
            'uncertainty_range': (mean_pred - 2*std_pred, mean_pred + 2*std_pred)
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
    """Example usage of Ensemble predictor"""
    # Paths to saved model and scaler
    model_path = 'saved_models/btc_ensemble_model.pkl'
    scaler_path = 'saved_models/ensemble_scaler.pkl'
    
    # Initialize predictor
    predictor = EnsemblePredictor(model_path, scaler_path)
    
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
        print(f"Ensemble predicted next Bitcoin price: ${next_price:.2f}")
        
        # Get individual model predictions
        print("\nGetting individual model predictions...")
        individual_preds = predictor.get_individual_predictions(recent_data)
        for model_name, pred in individual_preds.items():
            print(f"{model_name.upper()}: ${pred:.2f}")
        
        # Make prediction with uncertainty
        print("\nMaking prediction with uncertainty estimation...")
        uncertainty_pred = predictor.predict_with_uncertainty(recent_data)
        print(f"Ensemble prediction: ${uncertainty_pred['ensemble_prediction']:.2f}")
        print(f"Individual models mean: ${uncertainty_pred['mean_individual']:.2f}")
        print(f"Individual models std: ${uncertainty_pred['std_individual']:.2f}")
        print(f"Uncertainty range: ${uncertainty_pred['uncertainty_range'][0]:.2f} - ${uncertainty_pred['uncertainty_range'][1]:.2f}")
        
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