import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf

from data.data import get_data
from classes.model_data_class import ModelData

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Volume', 'Direction']
START_DATE = "1900-01-01"
END_DATE = "2024-02-01"


# SEQUENCE_COLUMNS = ['Close', 'Change', 'Direction']
# OUTPUT_COLUMNS = ['Close', 'Change', 'Direction']
# 'Stochastic_K', 'Stochastic_D',
# 'lag_1', 'lag_2', 'lag_3', 'lag_4',
SEQUENCE_COLUMNS = ['Open', 'Close', 'Change', 'RSI', 'MA25', 'MA50']
OUTPUT_COLUMNS = ['Change']

SEQUENCE_LENGTH = 1
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

    ma25 = ta.sma(data['Close'], length=25)
    extended_data['MA25'] = data['Close'] - ma25
    
    ma50 = ta.sma(data['Close'], length=50)
    extended_data['MA50'] = data['Close'] - ma50
    # extended_data['MA50'] = ta.sma(data['Close'], length=50)
    # extended_data['MA100'] = ta.sma(data['Close'], length=100)

    extended_data['RSI'] = ta.rsi(data['Close'])
    
    # stoch_results  = ta.stoch(high=data['High'], low=data['Low'], close=data['Close'])
    # extended_data['Stochastic_K'] = stoch_results.iloc[:, 0]
    # extended_data['Stochastic_D'] = stoch_results.iloc[:, 1]

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

    target = data['Direction'].to_numpy()
    scaled_data = scaled_data[SEQUENCE_COLUMNS].to_numpy()

    return scaled_data, target


def prepare_sequences(data: pd.DataFrame, target):
    indicators = data[SEQUENCE_COLUMNS]

    # Shift the target column to represent the next day's direction
    shifted_target = data['Direction'].shift(-1)
    
    # Remove the last row as it will have NaN due to shifting
    indicators = indicators[:-1]
    shifted_target = shifted_target[:-1]
    
    # Convert shifted target to numpy array
    target = shifted_target.to_numpy()
    return indicators, target


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


def get_cnn_data(ticker): 
    data = prepare_data(ticker)
    data = add_indicators(data)
    # data = add_lags(data)
    # indicators_train, indicators_test, target_train, target_test = split_data(extended_data)
    data, target = normalize_data(data)
    # data, target = prepare_sequences(data, target)
    train, test, predict = split_train_and_test_data(data, target)

    # x_train, y_train = prepare_tensors(train['x'], train['y'])
    # x_test, y_test = prepare_tensors(test['x'], test['y'])
    # x_predict, y_predict = prepare_tensors(predict['x'], predict['y'])
    
    return train, test, predict