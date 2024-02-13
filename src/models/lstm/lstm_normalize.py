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

SEQUENCE_LENGTH = 10

def prepare_data():
    stock_data = get_SP500_data(START_DATE, END_DATE)
    stock_data = stock_data[COLUMNS_V1]
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    return stock_data


def normalize_data(data): 
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    return scaled_data


def prepare_sequences(data):
    x = []
    y = []
    dates = data[SEQUENCE_LENGTH:].index.to_numpy()

    for i in range(SEQUENCE_LENGTH, len(data)):
        y_today = data['Close'].iloc[i]

        x_previous_days = data[SEQUENCE_COLUMNS_V1].iloc[i-SEQUENCE_LENGTH:i].values.flatten()

        x.append(x_previous_days)
        y.append(y_today)   

    x = np.array(x)
    y = np.array(y)

    return x, y, dates


def split_train_and_test_data(x, y, dates):
    q_80 = int(len(dates) * .8)
    q_100 = int(len(dates))
    
    x_train, y_train, dates_train = x[0:q_80], y[0:q_80], dates[0:q_80]
    x_test, y_test, dates_test = x[q_80:q_100], y[q_80:q_100], dates[q_80:q_100]
    
    return x_train, x_test, y_train, y_test, dates_train, dates_test


def prepare_tensors(x, y):
    x = x.reshape(x.shape[0], x.shape[1], 1)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return x, y


def get_lstm_data(): 
    data = prepare_data()
    normalized_data = normalize_data(data)
    x, y, dates = prepare_sequences(normalized_data)
    x_train, x_test, y_train, y_test, dates_train, dates_test = split_train_and_test_data(x, y, dates)
    # x_train, y_train = prepare_tensors(x_train, y_train)
    # x_test, y_test = prepare_tensors(x_test, y_test)
    
    return x_train, x_test, y_train, y_test, dates_train, dates_test