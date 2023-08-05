#Achieves 1.8-2.2 MSE for C
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from tensorflow.keras.optimizers import SGD
import yfinance as yf
import pandas as pd
import os

def get_historical_data(symbol):

    file_name = f"{symbol}_historical_data.csv"
    
    # Check if data file already exists locally
    if os.path.exists(file_name):
        print(f"File {file_name} found. Loading data from file.")
        data = pd.read_csv(file_name, index_col='Date', parse_dates=True)
        first_date = data.index[0]
        print(f"The first date of available data for {symbol} is: {first_date}")
    else:
        print(f"File {file_name} not found. Fetching data from Yahoo Finance.")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start='1900-01-01')
        
        if isinstance(data, pd.DataFrame) and not data.empty:
            first_date = data.index[0]
            print(f"The first date of available data for {symbol} is: {first_date}")
            # Save the data to a local file
            data['future_close']=data['Close'].shift(-1)
            data.dropna(inplace=True)
            data.to_csv(file_name)
        else:
            print("No data fetched for the symbol.")
            data = pd.DataFrame()  # return an empty DataFrame
            data['future_close']=data['Close'].shift(-1)
            data.dropna(inplace=True)
    return data




def preprocess_data(data):
    close_prices = data['future_close'].values
    features = data[['High', 'Low', 'Open', 'Close', 'Volume','Dividends','Stock Splits','future_close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)
    close_scale = close_prices.ptp()  # Peak to peak (max - min) value
    close_min = close_prices.min()
    joblib.dump((scaler, close_scale, close_min), 'scaler_and_close.pkl')
    return features, close_scale, close_min


def processData(data, lb):
    X, Y = [], []
    for i in range(len(data) - lb - 1):
        X.append(data[i: (i + lb), :-1])  # Exclude the 'future_close' column
        Y.append(data[(i + lb), -1])  # 'future_close' is the last column

    return np.array(X), np.array(Y)


def train_test_split(X, y, ratio):
    split = int(len(X) * ratio)
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    return X_train, X_test, y_train, y_test



def create_model(look_back): #4,64 128 128 128 best optimization
    encoder_inputs = keras.layers.Input(shape=(look_back, 7))
    encoder_x = MultiHeadAttention(num_heads=4, key_dim=16)(encoder_inputs, encoder_inputs)
    encoder_x = LayerNormalization(epsilon=1e-6)(encoder_x + encoder_inputs)

    encoder_outputs, state_h, state_c = keras.layers.LSTM(100, return_state=True)(encoder_x)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.layers.RepeatVector(1)(encoder_outputs)
    decoder_lstm = keras.layers.LSTM(100, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    decoder_outputs = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(decoder_outputs)
    decoder_outputs = keras.layers.Dense(100, activation="relu")(decoder_outputs)

    outputs = keras.layers.Dense(1, activation="linear")(decoder_outputs)
    optimizer = keras.optimizers.Adam(learning_rate=1e-6)
    #optimizer = SGD(learning_rate=0.0001)
    model = keras.models.Model(inputs=encoder_inputs, outputs=outputs)
    model.compile(loss="mse", optimizer=optimizer)
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    model.save('my_model')

def evaluate_model(model, X_test, y_test, close_scale, close_min):
    y_pred = model.predict(X_test)
    y_pred_rescaled = y_pred * close_scale + close_min
    y_test_rescaled = y_test * close_scale + close_min
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

def main(symbol):
    data = get_historical_data(symbol)
    features, close_scale, close_min = preprocess_data(data)
    look_back = 5
    X, y = processData(features, look_back)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)


    model = create_model(look_back)
    train_model(model, X_train, y_train, X_test, y_test, epochs=1000, batch_size=32)
    evaluate_model(model, X_test, y_test, close_scale, close_min)


def load_and_predict_last_ten_rows(symbol, close_scale, close_min):
    data = get_historical_data(symbol)
    features, _, _ = preprocess_data(data)
    look_back = 5
    X, y = processData(features, look_back)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)

    # Load the saved model
    model = keras.models.load_model('my_model')

    # Make predictions using the last 10 rows of X_test
    last_ten_rows = X_test[-10:]
    y_pred = model.predict(last_ten_rows)

    # Rescale the predictions and the actual close prices
    y_pred_rescaled = y_pred * close_scale + close_min
    y_test_rescaled = y_test[-10:] * close_scale + close_min

    print("\nLast 10 rows:")
    print("Actual Close Price | Predicted Close Price")
    for i in range(10):
        print(f"{y_test_rescaled[i]:.2f} | {y_pred_rescaled[i][0]:.2f}")


if __name__ == "__main__":
    symbol="c"
    main(symbol) # Comment this out if you don't want to re-train the model.
    scaler, close_scale, close_min = joblib.load('scaler_and_close.pkl')
    load_and_predict_last_ten_rows(symbol, close_scale, close_min)
