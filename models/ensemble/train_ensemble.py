import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_preprocessing import DataPreprocessor
from utils.visualization import DataVisualizer
from ensemble_model import EnsembleModel
from config.config import Config

def train_ensemble_model():
    """Complete ensemble model training pipeline"""
    print("Starting Ensemble Model Training...")
    print("="*50)
    
    # Initialize components
    preprocessor = DataPreprocessor(
        window_size=Config.WINDOW_SIZE,
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE
    )
    visualizer = DataVisualizer()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    try:
        df = preprocessor.load_and_clean_data(Config.RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"Raw data not found at {Config.RAW_DATA_PATH}")
        print("Trying processed data...")
        df = pd.read_csv(Config.PROCESSED_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # Prepare data for training
    data_dict = preprocessor.prepare_data(df, Config.TARGET_COLUMN)
    
    print(f"Data shapes:")
    print(f"X_train: {data_dict['X_train'].shape}")
    print(f"X_val: {data_dict['X_val'].shape}")
    print(f"X_test: {data_dict['X_test'].shape}")
    print(f"Input shape: {data_dict['input_shape']}")
    
    # Build and train ensemble model
    print("\nBuilding ensemble model...")
    ensemble_model = EnsembleModel(data_dict['input_shape'])
    ensemble_model.build_ensemble(epochs=20, batch_size=32)
    
    # Train the ensemble
    ensemble_model.train(data_dict['X_train'], data_dict['y_train'])
    
    # Evaluate the ensemble
    print("\nEvaluating ensemble model...")
    results = ensemble_model.evaluate(data_dict['X_test'], data_dict['y_test'])
    
    # Print results
    print(f"\n{'='*50}")
    print("ENSEMBLE MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"R² Score:      {results['r2_score']:.4f}")
    print(f"MSE:           {results['mse']:.6f}")
    print(f"RMSE:          {results['rmse']:.6f}")
    print(f"{'='*50}")
    
    # Plot predictions
    visualizer.plot_predictions(
        data_dict['y_test'], 
        results['predictions'],
        "Ensemble Predictions vs Actual"
    )
    
    # Compare individual model predictions if available
    try:
        # Get individual model predictions
        lstm_pred = ensemble_model.ensemble.named_estimators_['lstm'].predict(data_dict['X_test'])
        gru_pred = ensemble_model.ensemble.named_estimators_['gru'].predict(data_dict['X_test'])
        
        # Plot comparison
        plt.figure(figsize=(15, 8))
        
        sample_indices = np.random.choice(len(data_dict['y_test']), min(200, len(data_dict['y_test'])), replace=False)
        sample_indices = np.sort(sample_indices)
        
        plt.plot(sample_indices, data_dict['y_test'][sample_indices], label='Actual', alpha=0.8, linewidth=2)
        plt.plot(sample_indices, lstm_pred[sample_indices], label='LSTM', alpha=0.7, linestyle='--')
        plt.plot(sample_indices, gru_pred[sample_indices], label='GRU', alpha=0.7, linestyle='--')
        plt.plot(sample_indices, results['predictions'][sample_indices], label='Ensemble', alpha=0.8, linewidth=2)
        
        plt.title('Model Comparison: Individual vs Ensemble Predictions')
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except Exception as e:
        print(f"Could not create comparison plot: {e}")
    
    # Save model and scaler
    os.makedirs('saved_models', exist_ok=True)
    ensemble_model.save_model('saved_models/btc_ensemble_model.pkl')
    preprocessor.save_scaler('saved_models/ensemble_scaler.pkl')
    
    print("\nEnsemble training completed successfully!")
    return ensemble_model, results

if __name__ == "__main__":
    model, results = train_ensemble_model()
