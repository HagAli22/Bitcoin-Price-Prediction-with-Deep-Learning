import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import joblib
from utils.data_preprocessing import DataPreprocessor
from utils.evaluation import ModelEvaluator
from utils.visualization import DataVisualizer
from lstm_model import LSTMModel
from config.config import Config

def train_lstm_model():
    """Complete LSTM model training pipeline"""
    print("Starting LSTM Model Training...")
    print("="*50)
    
    # Initialize components
    preprocessor = DataPreprocessor(
        window_size=Config.WINDOW_SIZE,
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE
    )
    evaluator = ModelEvaluator()
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
    
    # Build LSTM model
    print("\nBuilding LSTM model...")
    lstm_model = LSTMModel(data_dict['input_shape'])
    model = lstm_model.build_model()
    
    print("\nModel Architecture:")
    lstm_model.get_model_summary()
    
    # Train and evaluate model
    print("\nTraining LSTM model...")
    results = evaluator.evaluate_model(
        model,
        data_dict['X_train'], data_dict['y_train'],
        data_dict['X_val'], data_dict['y_val'],
        data_dict['X_test'], data_dict['y_test'],
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )
    
    # Print results
    evaluator.print_evaluation_results(results, "LSTM")
    
    # Plot training history
    evaluator.plot_training_history(results['history'], "LSTM Model")
    
    # Plot predictions
    visualizer.plot_predictions(
        data_dict['y_test'], 
        results['predictions']['y_pred_test'].flatten(),
        "LSTM Predictions vs Actual"
    )
    
    # Save model and scaler
    os.makedirs('saved_models', exist_ok=True)
    lstm_model.save_model('saved_models/btc_lstm_model.h5')
    preprocessor.save_scaler('saved_models/lstm_scaler.pkl')
    
    print("\nTraining completed successfully!")
    return model, results

if __name__ == "__main__":
    model, results = train_lstm_model()
