import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data.sp500 import get_SP500_data
# import pandas_ta as ta

START_DATE = "2000-01-01"
END_DATE = "2024-02-01"
COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close']

sequence_length = 5 

def prepare_data():
    stock_data = get_SP500_data(START_DATE, END_DATE)
    stock_data = stock_data[COLUMNS]
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    return stock_data


def normalize_data(data): 
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    return scaled_data


def prepare_sequences(data):
    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X_today = data['Close'].iloc[i]

        y_previous_days = data[['Open', 'High', 'Low', 'Close']].iloc[i-sequence_length:i].values.flatten()

        X.append(X_today)
        y.append(y_previous_days)   

    X = np.array(X)
    y = np.array(y)

    return X, y


def prepare_tensors(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(X_train)

    X_train = X_train.reshape(X_train.shape[0], 1, 1)
    X_test = X_test.reshape(X_test.shape[0], 1, 1)

    # X_train = X_train[:, 1:]
    # X_test = X_test[:, 1:]
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    return X_train, X_test, y_train, y_test