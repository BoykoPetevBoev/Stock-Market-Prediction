import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data.sp500 import get_SP500_data
import pandas_ta as ta

START_DATE = "2000-01-01"
END_DATE = "2024-02-01"

COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close']
SEQUENCE_COLUMNS = ['Open', 'High', 'Low', 'Close']

SEQUENCE_LENGTH = 15
OUTPUT_LENGTH = 3

def prepare_data():
    stock_data = get_SP500_data(START_DATE, END_DATE)
    stock_data = stock_data[COLUMNS]
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    return stock_data

def add_indicators(data): 
    extended_data = data
    # extended_data['SMA'] = ta.sma(data['Close'], length=50)  # Simple Moving Average
    # extended_data['RSI'] = ta.rsi(data['Close'])
    # extended_data.dropna(inplace=True)
    return extended_data

def normalize_data(data): 
    matrix = np.array(data)
    original_shape = matrix.shape

    flattened_array = matrix.flatten()
    column_vector = flattened_array.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    scaled_data = scaler.fit_transform(column_vector)

    scaled_data = scaled_data.reshape(original_shape)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    return scaled_data


def prepare_sequences(data):
    x = []
    y = []
    x_dates = []
    y_dates = []

    for i in range(SEQUENCE_LENGTH, len(data) - OUTPUT_LENGTH - 1):
        y_target_days = data[SEQUENCE_COLUMNS].iloc[i + 1:i+OUTPUT_LENGTH+1].to_numpy()
        y_target_dates = data.index[i+1:i+OUTPUT_LENGTH+1].to_numpy()
        
        x_previous_days = data[SEQUENCE_COLUMNS].iloc[i-SEQUENCE_LENGTH+1:i+1].to_numpy()
        x_previous_dates = data.index[i-SEQUENCE_LENGTH+1:i+1].to_numpy()

        x.append(x_previous_days)
        y.append(y_target_days)
        y_dates.append(y_target_dates)
        x_dates.append(x_previous_dates)

    x = np.array(x)
    y = np.array(y)
    x_dates = np.array(x_dates)
    y_dates = np.array(y_dates)

    return x, y, x_dates, y_dates


def split_train_and_test_data(x, y, x_dates, y_dates):
    train = int(len(y_dates) * .8)
    test = int(len(y_dates) * .9)
    predict = int(len(y_dates))
    
    x_train, y_train = x[0:train], y[0:train]
    x_test, y_test = x[train:test], y[train:test]
    x_predict, y_predict = x[test:predict], y[test:predict]
    
    return {
        'x': x[0:train],
        'y': y[0:train],
        'x_dates': x_dates[0:train],
        'y_dates': y_dates[0:train]
    }, {
        'x': x_test,
        'y': y_test,
        'x_dates': x_dates[train:test],
        'y_dates': y_dates[train:test]
    }, {
        'x': x_predict,
        'y': y_predict,
        'x_dates': x_dates[test:predict],
        'y_dates': y_dates[test:predict]
    }


def prepare_tensors(x, y):
    x = x.reshape(x.shape[0], x.shape[1], 1)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return x, y


def get_lstm_data(): 
    data = prepare_data()
    normalized_data = normalize_data(data)
    x, y, x_dates, y_dates = prepare_sequences(normalized_data)
    train, test, predict = split_train_and_test_data(x, y, x_dates, y_dates)
    # x_train, y_train = prepare_tensors(train['x'], train['y'])
    # x_test, y_test = prepare_tensors(test['x'], test['y'])
    # x_predict, y_predict = prepare_tensors(predict['x'], predict['y'])
    
    return {
        'x': train['x'],
        'y': train['y'],
        'x_dates': train['x_dates'],
        'y_dates': train['y_dates']
    }, {
        'x': test['x'],
        'y': test['y'],
        'x_dates': test['x_dates'],
        'y_dates': test['y_dates'],
    }, {
        'x': predict['x'],
        'y': predict['y'],
        'x_dates': predict['x_dates'],
        'y_dates': predict['y_dates'],
    }
