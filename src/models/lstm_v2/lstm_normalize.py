import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf

from data.data import get_data
from classes.model_data_class import ModelData

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Volume']
START_DATE = "1900-01-01"
END_DATE = "2024-02-01"


# SEQUENCE_COLUMNS = ['Close', 'Change', 'Direction']
# OUTPUT_COLUMNS = ['Close', 'Change', 'Direction']
SEQUENCE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Change', 'Volume']
OUTPUT_COLUMNS = ['Change']

SEQUENCE_LENGTH = 5
OUTPUT_LENGTH = 1

def prepare_data(ticker: str):
    stock_data = get_data(
        ticker = ticker, 
        start_date = START_DATE, 
        end_date = END_DATE
    )
    stock_data = stock_data[COLUMNS]
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    # stock_data["Year"] = stock_data.Date.dt.year
    # stock_data["Month"] = stock_data.Date.dt.month
    # stock_data["Day"] = stock_data.Date.dt.day
    stock_data = stock_data.drop(columns=["Date"])
    # stock_data.set_index('Date', inplace=True)
    # stock_data.reset_index()
    return stock_data


def add_indicators(data: pd.DataFrame): 
    extended_data = data
    extended_data['MA25'] = ta.sma(data['Close'], length=25)
    extended_data['MA50'] = ta.sma(data['Close'], length=50)
    extended_data['MA100'] = ta.sma(data['Close'], length=100)
    extended_data['RSI'] = ta.rsi(data['Close'])
    extended_data.dropna(inplace=True)
    extended_data.reset_index()
    return extended_data


def add_lags(data: pd.DataFrame):
    for lag in range(1, 5):
        data[f"lag_{lag}"] = data.Change.shift(lag)

    data = data.dropna()
    data = data.reset_index()
    return data

def split_data(data: pd.DataFrame):
    target = data.Change
    indicators = data.drop(columns=["Change"])
    indicators_train, indicators_test, target_train, target_test =  train_test_split(indicators, target, test_size=0.2)
    return indicators_train, indicators_test, target_train, target_test


def normalize_data(data: pd.DataFrame): 
    # matrix = np.array(data)
    # original_shape = matrix.shape

    # flattened_array = matrix.flatten()
    # column_vector = flattened_array.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # scaled_data = scaled_data.reshape(original_shape)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    return scaled_data


def prepare_sequences(data: pd.DataFrame):
    x = []
    y = []
    # x_dates = []
    # y_dates = []

    target = (data['Change'] > 0).astype(int)
    indicators = data

    for i in range(SEQUENCE_LENGTH, len(data)):
        y_target_days = target.iloc[i]
    #     y_target_dates = data.index[i+1:i+OUTPUT_LENGTH+1].to_numpy()
        
        x_previous_days = indicators.iloc[i-SEQUENCE_LENGTH:i]
    #     x_previous_dates = data.index[i-SEQUENCE_LENGTH+1:i+1].to_numpy()

        x.append(x_previous_days)
        y.append(y_target_days)
    #     y_dates.append(y_target_dates)
    #     x_dates.append(x_previous_dates)

    x = np.array(x)
    y = np.array(y)
    # x_dates = np.array(x_dates)
    # y_dates = np.array(y_dates)

    return x, y


def split_train_and_test_data(x, y):
    train = int(len(y) * .8)
    test = int(len(y) * .9)
    predict = int(len(y))
    
    x_train, y_train = x[0:train], y[0:train]
    x_test, y_test = x[train:test], y[train:test]
    x_predict, y_predict = x[test:predict], y[test:predict]
    
    return {
        'x': x[0:train],
        'y': y[0:train],
        # 'x_dates': x_dates[0:train],
        # 'y_dates': y_dates[0:train]
    }, {
        'x': x_test,
        'y': y_test,
        # 'x_dates': x_dates[train:test],
        # 'y_dates': y_dates[train:test]
    }, {
        'x': x_predict,
        'y': y_predict,
        # 'x_dates': x_dates[test:predict],
        # 'y_dates': y_dates[test:predict]
    }


def prepare_tensors(x, y):
    x = x.reshape(x.shape[0], x.shape[1], 1)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return x, y


def get_lstm_data(ticker): 
    data = prepare_data(ticker)
    # extended_data = add_indicators(data)
    # extended_data = add_lags(extended_data)
    # indicators_train, indicators_test, target_train, target_test = split_data(extended_data)
    x, y = prepare_sequences(data)
    train, test, predict = split_train_and_test_data(x, y)

    # normalized_data = normalize_data(data)
    # x_train, y_train = prepare_tensors(train['x'], train['y'])
    # x_test, y_test = prepare_tensors(test['x'], test['y'])
    # x_predict, y_predict = prepare_tensors(predict['x'], predict['y'])
    
    return train, test, predict
