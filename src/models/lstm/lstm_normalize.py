import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data.sp500 import get_SP500_data
# import pandas_ta as ta

START_DATE = "2000-01-01"
END_DATE = "2024-02-01"

COLUMNS_V1 = ['Date', 'Close']
SEQUENCE_COLUMNS_V1 = ['Close']

COLUMNS_V2 = ['Date', 'Open', 'High', 'Low', 'Close']
SEQUENCE_COLUMNS_V2 = ['Open', 'High', 'Low', 'Close']

sequence_length = 5 

def prepare_data():
    stock_data = get_SP500_data(START_DATE, END_DATE)
    stock_data = stock_data[COLUMNS_V1]
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    return stock_data


def normalize_data(data): 
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    return scaled_data


def prepare_sequences(data):
    x = []
    y = []

    for i in range(sequence_length, len(data)):
        y_today = data['Close'].iloc[i]

        x_previous_days = data[SEQUENCE_COLUMNS_V1].iloc[i-sequence_length:i].values.flatten()

        x.append(x_previous_days)
        y.append(y_today)   

    x = np.array(x)
    y = np.array(y)

    return x, y


def split_train_and_test_data(x, y):
     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
     return x_train, x_test, y_train, y_test


def prepare_tensors(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    return x_train, x_test, y_train, y_test


def get_lstm_data(): 
    data = prepare_data()
    normalized_data = normalize_data(data)
    x, y = prepare_sequences(normalized_data)
    x_train, x_test, y_train, y_test = prepare_tensors(x, y)
    
    return x_train, x_test, y_train, y_test