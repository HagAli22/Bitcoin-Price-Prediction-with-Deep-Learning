class Config:
    # Data parameters
    WINDOW_SIZE = 30
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    TARGET_COLUMN = 'close'
    
    # Training parameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # File paths
    RAW_DATA_PATH = 'data/raw/BTC_USDT_ohlcv_data.parquet'
    PROCESSED_DATA_PATH = 'data/processed/BTC_USDT_ohlcv_data.csv'
    SCALER_PATH = 'models/scaler.pkl'
    
    # Model parameters
    CNN_PARAMS = {
        'conv1_filters': 128,
        'conv1_kernel': 3,
        'conv2_filters': 256,
        'conv2_kernel': 3,
        'dense_units': 128,
        'dropout_rates': [0.2, 0.3, 0.5]
    }
    
    LSTM_PARAMS = {
        'lstm1_units': 100,
        'lstm2_units': 100,
        'dense_units': 128,
        'dropout_rates': [0.2, 0.3, 0.5]
    }
    
    GRU_PARAMS = {
        'gru1_units': 100,
        'gru2_units': 100,
        'dense_units': 128,
        'dropout_rates': [0.2, 0.3, 0.4]
    }