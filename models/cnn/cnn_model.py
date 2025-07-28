import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import Config

class CNNModel:
    def __init__(self, input_shape, config=None):
        self.input_shape = input_shape
        self.config = config or Config.CNN_PARAMS
        self.model = None
        
    def build_model(self):
        """Build the CNN model architecture"""
        model = Sequential([
            # First Convolutional Block
            Conv1D(
                self.config['conv1_filters'], 
                kernel_size=self.config['conv1_kernel'], 
                activation='relu', 
                input_shape=self.input_shape
            ),
            MaxPooling1D(pool_size=2),
            Dropout(self.config['dropout_rates'][0]),
            
            # Second Convolutional Block
            Conv1D(
                self.config['conv2_filters'], 
                kernel_size=self.config['conv2_kernel'], 
                activation='relu'
            ),
            MaxPooling1D(pool_size=2),
            Dropout(self.config['dropout_rates'][1]),
            
            # Dense Layers
            Flatten(),
            Dense(self.config['dense_units'], activation='relu'),
            Dropout(self.config['dropout_rates'][2]),
            Dense(1)  # Output layer
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss=MeanSquaredError()
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