import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError



# Or if it is in a module, import it:
# from my_model_utils import build_lstm

# Loading scaleres and model
def load_resources():   
    CNN_model = joblib.load('btc_cnn_model.pkl')
    LSTM_model  = load_model('btc_lstm_model.h5', custom_objects={'mse': MeanSquaredError()})
    GRU_model = load_model('btc_gru_model.h5', custom_objects={'mse': MeanSquaredError()})
    #V_model = joblib.load('btc_voting_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return CNN_model,LSTM_model,GRU_model,   scaler

def main():
    # title and header
    st.title('💲Bitcoin Price Predection App')
    

    # Load resources
    CNN_model,LSTM_model, GRU_model, scaler = load_resources()

    # Take user input
    input_data = {}

    default_date = pd.to_datetime('2021-01-01')
    input_data['date'] = st.date_input("Select a date", value=default_date)
    input_data['open'] = st.number_input("Open Price", min_value=0.0, step=0.01)
    input_data['high'] = st.number_input("High Price", min_value=0.0, step=0.01)
    input_data['low'] = st.number_input("Low Price", min_value=0.0, step=0.01)
    input_data['volume'] = st.number_input("Volume", min_value=0.0, step=0.01)
    

    # Prediction
    if st.button('Predict'):
        # Create a DataFrame with user input
# Create a DataFrame with user input
        df_input = pd.DataFrame([{
        'open': input_data['open'],
        'high': input_data['high'],
        'low': input_data['low'],
        'close': 0.0,
        'volume': input_data['volume']
    }], index=[pd.to_datetime(input_data['date'])])
        df_input.index.name = 'date'

        # Scale the numerical columns
        scaled_array = scaler.transform(df_input)

        # Convert the scaled array back to DataFrame
        scaled_df = pd.DataFrame(scaled_array, columns=df_input.columns, index=df_input.index)

        # Drop the 'close' column from the scaled DataFrame
        scaled_df.drop('close', axis=1, inplace=True)

        # Set input_df to the cleaned, scaled data
        input_df = scaled_df

        # Prepare windowed sequence (repeat input for CNN shape)
        window_size = 30
        X_sequence = np.tile(input_df.values, (window_size, 1)).reshape((1, window_size, input_df.shape[1]))


        # Predict with each model
        pred_cnn = CNN_model.predict(X_sequence)[0][0]
        pred_lstm = LSTM_model.predict(X_sequence)[0][0]
        pred_gru = GRU_model.predict(X_sequence)[0][0]
        #pred_voting = V_model.predict(X_sequence)[0][0]

        
        pred_cnn_rescaled = scaler.inverse_transform([[input_data['open'], input_data['high'], input_data['low'], input_data['volume'], pred_cnn]])[0][-1]
        pred_lstm_rescaled = scaler.inverse_transform([[input_data['open'], input_data['high'], input_data['low'], input_data['volume'], pred_lstm]])[0][-1]
        pred_gru_rescaled = scaler.inverse_transform([[input_data['open'], input_data['high'], input_data['low'], input_data['volume'], pred_gru]])[0][-1]
        #pred_voting_rescaled = scaler.inverse_transform([[input_data['open'], input_data['high'], input_data['low'], input_data['volume'], pred_voting]])[0][-1]

        # Display the predictionsٍ
        st.write(f"📊 CNN Model Prediction for Close Price: {pred_cnn_rescaled} 💵")
        st.write(f"📈 LSTM Model Prediction for Close Price: {pred_lstm_rescaled} 💰")
        st.write(f"📉 GRU Model Prediction for Close Price: {pred_gru_rescaled} 💲")
        #st.write(f"🤖 Voting Model Prediction for Close Price: {pred_voting_rescaled} 💸")
# Run the app
if __name__ == '__main__':
    main()
