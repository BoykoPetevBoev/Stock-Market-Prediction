import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from data.data import get_data
from classes.model_data_class import ModelData

START_DATE = "1900-01-01"
END_DATE = "2024-02-01"

COLUMNS = ['Date', 'Close']
SEQUENCE_COLUMNS = ['Close']

SEQUENCE_LENGTH = 10

def prepare_data(ticker: str):
    stock_data = get_data(ticker, start_date=START_DATE, end_date=END_DATE)
    stock_data = stock_data[COLUMNS]
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    
    return stock_data


def normalize_data(data: pd.DataFrame): 
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    return scaled_data


def prepare_sequences(data: pd.DataFrame):
    x = []
    y = []
    dates = data[SEQUENCE_LENGTH:].index.to_numpy()

    for i in range(SEQUENCE_LENGTH, len(data)):
        y_today = data[SEQUENCE_COLUMNS].iloc[i]
        x_previous_days = data[SEQUENCE_COLUMNS].iloc[i-SEQUENCE_LENGTH:i].values.flatten()
        x.append(x_previous_days)
        y.append(y_today)   

    sequences = ModelData(x, y, dates)
    return sequences


def split_train_and_test_data(data: ModelData):
    train = int(len(data.dates) * .8)
    test = int(len(data.dates) * .9)
    predict = int(len(data.dates))
    
    train_data = ModelData(
        data.x[0:train], 
        data.y[0:train], 
        data.dates[0:train]
    )
    test_data = ModelData(
        data.x[train:test], 
        data.y[train:test], 
        data.dates[train:test]
    )
    predict_data = ModelData(
        data.x[test:predict], 
        data.y[test:predict], 
        data.dates[test:predict]
    )
    return train_data, test_data, predict_data


def prepare_tensors(x, y):
    x = x.reshape(x.shape[0], x.shape[1], 1)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return x, y


def get_lstm_data(ticker: str): 
    data = prepare_data(ticker)
    normalized_data = normalize_data(data)
    sequences_data = prepare_sequences(normalized_data)
    train, test, predict = split_train_and_test_data(sequences_data)
    return train, test, predict