import numpy as np
import pandas as pd
import tensorflow as tf

from data.data import get_data
from classes.model_data_class import ModelData
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


START_DATE = "1900-01-01"
END_DATE = "2024-02-01"

SEQUENCE_COLUMNS = ['Close', 'MA10', 'MA20']
OUTPUT_COLUMNS = ['Change']

SEQUENCE_LENGTH = 10
OUTPUT_LENGTH = 1

BEARISH = 0
NEUTRAL = 1
BULLISH = 2

TREND_CLASSES = ["Bearish", "Neutral", "Bullish"]


def calclate_sort_term_trend(data):
    data['Short_Term_Trend'] = NEUTRAL

    upward_condition = (data['Close'] > data['MA20']) & (data['MA10'] > data['MA20'])
    data.loc[upward_condition, 'Short_Term_Trend'] = BULLISH

    downward_condition = (data['Close'] < data['MA20']) & (data['MA10'] < data['MA20'])
    data.loc[downward_condition, 'Short_Term_Trend'] = BEARISH

    return data


def calclate_long_term_trend(data):
    data['Long_Term_Trend'] = NEUTRAL
    data.loc[(data['Close'] > data['MA100']) & (data['MA50'] > data['MA100']), 'Long_Term_Trend'] = BULLISH
    data.loc[(data['Close'] < data['MA100']) & (data['MA50'] < data['MA100']), 'Long_Term_Trend'] = BEARISH
    return data
    

def prepare_data(ticker: str):
    stock_data = get_data(
        ticker = ticker, 
        start_date = START_DATE, 
        end_date = END_DATE
    )
    
    stock_data = calclate_sort_term_trend(stock_data)
    stock_data = calclate_long_term_trend(stock_data)
    
    target = stock_data['Short_Term_Trend'].to_numpy()
    stock_data.reset_index(inplace=True)
    return stock_data, target


def add_lags(data: pd.DataFrame):
    for lag in range(1, 5):
        data[f"lag_{lag}"] = data.Change.shift(lag)

    data = data.dropna()
    return data


def split_data(data: pd.DataFrame):
    target = [data.Short_Term_Trend, data.Long_Term_Trend]
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


def prepare_sequences(data: pd.DataFrame, target):
    x = []
    y = []
    x_dates = []

    dates = data["Date"]
    indicators = data[SEQUENCE_COLUMNS]

    for i in range(SEQUENCE_LENGTH, len(indicators)):
        
        x_previous_days = indicators.iloc[i-SEQUENCE_LENGTH:i].to_numpy()
        rounded_data = np.round(x_previous_days, decimals=3)
        x.append(rounded_data)

        y.append(target[i - 1])

        dates_range = dates.iloc[i-SEQUENCE_LENGTH:i].to_numpy()
        x_dates.append(dates_range)

    x = np.array(x)
    y = np.array(y)
    x_dates = np.array(x_dates)

    return x, y, x_dates


def split_train_and_test_data(x, y, dates):
    train = int(len(y) * .8)
    test = int(len(y) * .9)
    predict = int(len(y))
    
    x_train, y_train = x[0:train], y[0:train]
    x_test, y_test = x[train:test], y[train:test]
    x_predict, y_predict = x[test:predict], y[test:predict]
    
    return {
        'x': x[0:train],
        'y': y[0:train],
        'dates': dates[0:train],
    }, {
        'x': x_test,
        'y': y_test,
        'dates': dates[train:test],
    }, {
        'x': x_predict,
        'y': y_predict,
        'dates': dates[test:predict],
    }


def prepare_tensors(x, y):
    x = x.reshape(x.shape[0], x.shape[1], 1)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return x, y


def get_lstm_data(ticker): 
    data, target = prepare_data(ticker)
    # data = add_lags(data)
    # indicators_train, indicators_test, target_train, target_test = split_data(extended_data)
    # data = normalize_data(data)
    data, target, dates = prepare_sequences(data, target)
    train, test, predict = split_train_and_test_data(data, target, dates)

    return train, test, predict
