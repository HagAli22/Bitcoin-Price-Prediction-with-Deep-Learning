import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import Config

class GRUModel:
    def __init__(self, input_shape, config=None):
        self.input_shape = input_shape
        self.config = config or Config.GRU_PARAMS
        self.model = None
        
    def build_model(self):
        """Build the GRU model architecture"""
        model = Sequential([
            # First GRU Layer
            GRU(
                self.config['gru1_units'], 
                activation='relu', 
                return_sequences=True, 
                input_shape=self.input_shape
            ),
            Dropout(self.config['dropout_rates'][0]),
            
            # Second GRU Layer
            GRU(
                self.config['gru2_units'], 
                activation='relu', 
                return_sequences=False
            ),
            Dropout(self.config['dropout_rates'][1]),
            
            # Dense Layers
            Dense(self.config['dense_units'], activation='relu'),
            Dropout(self.config['dropout_rates'][2]),
            Dense(1)  # Output layer
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='mse'
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build and train the model first.")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model