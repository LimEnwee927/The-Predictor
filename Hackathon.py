import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Concatenate, LSTM, Bidirectional, BatchNormalization, Conv1D, MaxPooling1D, Flatten # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler # type: ignore
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from PIL import Image

# Function to calculate RSI
def compute_rsi(prices, period=14):
 delta = prices.diff()
 gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
 loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
 rs = gain / loss
 rsi = 100 - (100 / (1 + rs))
 return rsi

# Function to calculate SMA
def compute_sma(prices, period):
 return prices.rolling(window=period).mean()

# Function to calculate MACD
def compute_macd(prices, short_window=12, long_window=26, signal_window=9):
 short_ema = prices.ewm(span=short_window, adjust=False).mean()
 long_ema = prices.ewm(span=long_window, adjust=False).mean()
 macd = short_ema - long_ema
 signal = macd.ewm(span=signal_window, adjust=False).mean()
 return macd, signal

# Function to calculate MACD for future predictions
def compute_macd_future(prices, short_window=12, long_window=26, signal_window=9):
 short_ema = prices.ewm(span=short_window, adjust=False).mean()
 long_ema = prices.ewm(span=long_window, adjust=False).mean()
 macd = short_ema - long_ema
 signal = macd.ewm(span=signal_window, adjust=False).mean()
 return macd, signal

# Function to calculate ATR for future predictions
def compute_atr_future(high, low, close, period=14):
 high_low = high - low
 high_close = np.abs(high - close.shift())
 low_close = np.abs(low - close.shift())
 true_range = np.maximum(high_low, high_close, low_close)
 atr = true_range.rolling(window=period).mean()
 return atr

# Function to preprocess data
def preprocess_data(ticker, start_date, end_date, seq_length=60, sma_period=50, rsi_period=14):
 # Step 1: Fetch data using yfinance
 try:
     data = yf.download(ticker, start=start_date, end=end_date)
     print("Data fetched successfully!")
     print(data.head())  # Print the first few rows to verify
 except Exception as e:
     print(f"Error fetching data: {e}")
     return None, None, None, None, None, None

 # Ensure the data has the necessary columns
 if 'Close' not in data.columns:
     print("Column 'Close' not found. Available columns:", data.columns)
     return None, None, None, None, None, None

 # Calculate technical indicators
 data['SMA'] = compute_sma(data['Close'], sma_period)
 data['RSI'] = compute_rsi(data['Close'], rsi_period)
 data['MACD'], data['MACD_Signal'] = compute_macd(data['Close'])

 # Handle missing values
 data.ffill(inplace=True)  # Forward fill
 data.bfill(inplace=True)  # Backward fill

 # Normalize data
 scaler = MinMaxScaler(feature_range=(0, 1))
 scaled_data = scaler.fit_transform(data[['Close', 'Open', 'High', 'Low', 'Volume', 'SMA', 'RSI', 'MACD', 'MACD_Signal']])

 # Create sequences
 def create_sequences(data, seq_length):
     X, y = [], []
     for i in range(len(data) - seq_length):
         X.append(data[i:i+seq_length])
         y.append(data[i+seq_length, 0])  # Assuming 'Close' price is the target
     return np.array(X), np.array(y)


 X, y = create_sequences(scaled_data, seq_length)

 # Split data into training and testing sets
 train_size = int(len(X) * 0.8)
 X_train, X_test = X[:train_size], X[train_size:]
 y_train, y_test = y[:train_size], y[train_size:]

 # Reshape X_train and X_test for the model
 X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_train.shape[2]))

 return X_train, X_test, y_train, y_test, scaler, data

# Function to build the Transformer encoder
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
 # Multi-head self-attention
 attention_output = MultiHeadAttention(
     num_heads=num_heads, key_dim=head_size, dropout=dropout
 )(inputs, inputs)
  # Add & Normalize (Residual connection + Layer normalization)
 attention_output = Dropout(dropout)(attention_output)
 attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
  # Feed-forward network
 ff_output = Dense(ff_dim, activation="relu")(attention_output)
 ff_output = Dense(inputs.shape[-1])(ff_output)
  # Add & Normalize (Residual connection + Layer normalization)
 ff_output = Dropout(dropout)(ff_output)
 ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
 return ff_output

# Function to build the hybrid LSTM-Transformer-CNN model
def build_hybrid_model(input_shape, lstm_units=128, head_size=64, num_heads=4, ff_dim=128, num_layers=2, dropout=0.2):
 inputs = Input(shape=input_shape)
  # CNN branch
 cnn_output = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
 cnn_output = MaxPooling1D(pool_size=2)(cnn_output)
 cnn_output = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_output)
 cnn_output = MaxPooling1D(pool_size=2)(cnn_output)
 cnn_output = Flatten()(cnn_output)
 cnn_output = Dense(64, activation='relu')(cnn_output)
 cnn_output = Dropout(dropout)(cnn_output)
  # LSTM branch
 lstm_output = Bidirectional(LSTM(lstm_units, return_sequences=True, activation='tanh'))(inputs)
 lstm_output = Dropout(dropout)(lstm_output)
 lstm_output = BatchNormalization()(lstm_output)
  # Transformer branch
 x = inputs
 for _ in range(num_layers):
     x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
 transformer_output = GlobalAveragePooling1D()(x)
  # Concatenate CNN, LSTM, and Transformer outputs
 combined_output = Concatenate()([cnn_output, lstm_output[:, -1, :], transformer_output])
  # Fully connected layers
 x = Dense(64, activation='relu')(combined_output)
 x = Dropout(dropout)(x)
 x = Dense(32, activation='relu')(x)
 outputs = Dense(1)(x)  # Output layer for regression
 model = Model(inputs, outputs)
 return model

# Function to build and train the hybrid model
def build_and_train_model(X_train, y_train, X_test, y_test):
 # Define the hybrid model
 input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
 model = build_hybrid_model(
     input_shape=input_shape,
     lstm_units=128,  # Number of LSTM units
     head_size=64,    # Size of the attention head
     num_heads=4,     # Number of attention heads
     ff_dim=128,      # Hidden layer size in feed-forward network
     num_layers=2,    # Number of transformer encoder layers
     dropout=0.2      # Dropout rate
 )
  # Compile the model
 model.compile(optimizer='adam', loss='mean_absolute_error')

 # Early stopping to prevent overfitting
 early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

 # Learning rate scheduler
 def lr_scheduler(epoch, lr):
     return lr * 0.9 if epoch >= 10 else lr

 # Train the model
 history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test),
                     callbacks=[early_stopping, LearningRateScheduler(lr_scheduler)])

 return model, history

# Function to predict future prices with adaptive learning and noise injection
def predict_future(model, last_sequence, scaler, future_steps=7, noise_std=0.01):
 future_predictions = []
 future_high = []  # Simulated high prices
 future_low = []   # Simulated low prices
 current_sequence = last_sequence.copy()

 for _ in range(future_steps):
     # Predict the next time step
     next_prediction = model.predict(current_sequence[np.newaxis, :, :])
  
     # Add noise to simulate market volatility
     next_prediction += np.random.normal(0, noise_std, size=next_prediction.shape)
  
     # Simulate high and low prices (for ATR calculation)
     future_high.append(next_prediction[0, 0] + np.abs(np.random.normal(0, noise_std)))
     future_low.append(next_prediction[0, 0] - np.abs(np.random.normal(0, noise_std)))
  
     future_predictions.append(next_prediction[0, 0])

     # Update the sequence with the new prediction
     current_sequence = np.roll(current_sequence, -1, axis=0)
     current_sequence[-1, 0] = next_prediction  # Update the 'Close' price

 # Inverse transform the predictions
 future_predictions = np.array(future_predictions).reshape(-1, 1)
 dummy_array = np.zeros((len(future_predictions), scaler.n_features_in_))
 dummy_array[:, 0] = future_predictions.reshape(-1)
 future_predictions = scaler.inverse_transform(dummy_array)[:, 0]

 # Calculate MACD and ATR for future predictions
 future_prices = pd.Series(future_predictions)
 future_macd, future_signal = compute_macd_future(future_prices)
 future_atr = compute_atr_future(pd.Series(future_high), pd.Series(future_low), future_prices)
 return future_predictions, future_macd, future_signal, future_atr

# Function to generate buy/sell advice for future predictions
def generate_future_advice(future_predictions, future_macd, future_signal, future_atr):
 advice = []
 atr_threshold = future_atr.mean()  # Use mean ATR as a threshold
 for i in range(len(future_predictions)):
     if future_macd.iloc[i] > future_signal.iloc[i] and future_atr.iloc[i] < atr_threshold:
         advice.append("Buy in volume")
     elif future_macd.iloc[i] < future_signal.iloc[i] or future_atr.iloc[i] > atr_threshold:
         advice.append("Sell out all volume")
     else:
         advice.append("Hold")

 return advice

# Function to evaluate the model and visualize results
def evaluate_and_visualize(model, X_test, y_test, scaler, data):
 # Make predictions on the test data
 predictions = model.predict(X_test)
  # Inverse transform the predictions
 dummy_array_predictions = np.zeros((len(predictions), scaler.n_features_in_))
 dummy_array_predictions[:, 0] = predictions.reshape(-1)
 dummy_array_predictions = scaler.inverse_transform(dummy_array_predictions)
 predictions = dummy_array_predictions[:, 0]

 # Inverse transform the actual prices
 dummy_array_actual = np.zeros((len(y_test), scaler.n_features_in_))
 dummy_array_actual[:, 0] = y_test.reshape(-1)
 dummy_array_actual = scaler.inverse_transform(dummy_array_actual)
 actual_prices = dummy_array_actual[:, 0]

 # Predict future prices for the next 7 days
 last_sequence = X_test[-1]  # Use the last sequence in the test data
 future_predictions, future_macd, future_signal, future_atr = predict_future(model, last_sequence, scaler)

 # Generate buy/sell advice for future predictions
 future_advice = generate_future_advice(future_predictions, future_macd, future_signal, future_atr)
 test_dates = data.index[-len(y_test):]  # Dates for the test data

 # Print future predictions and advice
 future_dates = pd.date_range(start=data.index[-1], periods=8)[1:]
 st.write("\n7-Day Future Predictions and Advice:")
 for date, price, advice in zip(future_dates, future_predictions, future_advice):
     st.write(f"{date}: , Advice = {advice}")

 # Plot the results
 plt.figure(figsize=(14, 7))
 plt.plot(data.index[-len(y_test):], actual_prices, color='blue', label='Actual Prices')
 plt.plot(data.index[-len(y_test):], predictions, color='red', label='Predicted Prices (Test Data)')
 plt.plot(future_dates, future_predictions, color='green', label='7-Day Future Prediction')
 plt.title('Price Prediction with Future Buy/Sell Advice')
 plt.xlabel('Time')
 plt.ylabel('Price')
 plt.legend()
 plt.grid(True)
 plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
 plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
 plt.xticks(fontsize=3)
 plt.yticks(fontsize=3)
 plt.gcf().autofmt_xdate()
 plt.savefig('price_prediction_with_future_advice.png')
 st.pyplot(plt.gcf())

 plt.figure(figsize=(14, 7))
  # Select the last 5 days of test data and the future 7 days
 zoom_test_dates = test_dates[-5:]  # Last 5 days of test data
 zoom_test_predictions = predictions[-5:]  # Corresponding predictions
 zoom_future_dates = future_dates  # Future 7 days
 zoom_future_predictions = future_predictions  # Future predictions

 # Plot the zoomed-in data
 plt.plot(zoom_test_dates, zoom_test_predictions, color='red', label='Predicted Prices (Test Data)')
 plt.plot(zoom_future_dates, zoom_future_predictions, color='green', label='1-Week Future Prediction')
  # Add a vertical line to indicate the transition point
 transition_date = zoom_test_dates[-1]
 plt.axvline(x=transition_date, color='gray', linestyle='--', label='Transition Point')








 # Add titles and labels
 plt.title('Zoomed-In View: Transition Between Test and Future Predictions')
 plt.xlabel('Time')
 plt.ylabel('Price')
 plt.legend()
 plt.grid(True)
  # Format the x-axis to show dates properly
 plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
 plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Daily grid
 plt.gcf().autofmt_xdate()  # Rotate date labels
 plt.xticks(fontsize=8)
 plt.yticks(fontsize=8)
 # Set y-axis grid every 1000 units
 plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1000))
  # Save the plot as an image
 plt.savefig('hybrid_price_prediction_plot_zoomed.png')
  # Show the plot
 st.pyplot(plt.gcf())

# Function to save the trained model
def save_model(model, model_name="hybrid_market_model.h5"):
 model.save(model_name)
 print(f"Model saved as {model_name}")

# Main function
def main():
    st.title("Cryptocurrency Price Prediction and Trading Advice")
    st.write("This app predicts future cryptocurrency prices and provides buy/sell advice based on a hybrid LSTM-Transformer-CNN model.")
 

    # Display the image
    image_path = "algo.jpeg"  # Make sure the image is in the same folder as your streamline.py
    image = Image.open(image_path)
    st.image(image, caption="Cryptocurrency Market", use_column_width=True)

    # User inputs
    ticker = st.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "DOGE-USD", "LTC-USD", "XRP-USD", "ADA-USD"])
    start_date = st.date_input("Start Date for Training Data", pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date for Training Data", pd.to_datetime("2023-01-01"))

    if st.button("Train Model and Predict"):
        st.write("Training the model... This may take a few minutes.")
        
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler, data = preprocess_data(ticker, start_date, end_date)

        if X_train is None:
            st.error("Error: Unable to fetch or preprocess data. Please check your inputs.")
            return

        # Build and train the model
        model, history = build_and_train_model(X_train, y_train, X_test, y_test)

        # Evaluate and visualize results
        st.write("Model training complete. Generating predictions...")
        evaluate_and_visualize(model, X_test, y_test, scaler, data)

# Run the main function
if __name__ == "__main__":
 main()




