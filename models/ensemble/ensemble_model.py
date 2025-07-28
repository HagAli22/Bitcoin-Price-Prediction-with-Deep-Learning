import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import Config

def create_lstm_model(input_shape):
    """Create LSTM model for ensemble"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(100, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_gru_model(input_shape):
    """Create GRU model for ensemble"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        GRU(100, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(100, activation='relu', return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

class EnsembleModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.ensemble = None
        self.scaler = None
        
    def build_ensemble(self, epochs=20, batch_size=32):
        """Build voting ensemble with LSTM and GRU models"""
        # Create individual model wrappers
        lstm_reg = KerasRegressor(
            model=create_lstm_model,
            model__input_shape=self.input_shape,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        gru_reg = KerasRegressor(
            model=create_gru_model,
            model__input_shape=self.input_shape,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Create voting ensemble
        self.ensemble = VotingRegressor(
            estimators=[
                ('lstm', lstm_reg),
                ('gru', gru_reg)
            ]
        )
        
        return self.ensemble
    
    def train(self, X_train, y_train):
        """Train the ensemble model"""
        if self.ensemble is None:
            self.build_ensemble()
        
        print("Training ensemble model...")
        self.ensemble.fit(X_train, y_train)
        print("Ensemble training completed!")
        
    def predict(self, X):
        """Make predictions using the ensemble"""
        if self.ensemble is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.ensemble.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the ensemble model"""
        predictions = self.predict(X_test)
        
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'predictions': predictions
        }
    
    def save_model(self, filepath):
        """Save the trained ensemble model"""
        if self.ensemble is not None:
            joblib.dump(self.ensemble, filepath)
            print(f"Ensemble model saved to {filepath}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, filepath):
        """Load a saved ensemble model"""
        self.ensemble = joblib.load(filepath)
        print(f"Ensemble model loaded from {filepath}")
        return self.ensemble
