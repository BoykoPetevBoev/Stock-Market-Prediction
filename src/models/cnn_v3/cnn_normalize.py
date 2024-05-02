import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf

from data.data import get_data
from classes.model_data_class import ModelData

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


START_DATE = "1900-01-01"
END_DATE = "2024-02-01"

SEQUENCE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Direction']
OUTPUT_COLUMNS = ['Change']

SEQUENCE_LENGTH = 1
OUTPUT_LENGTH = 1


def is_shooting_star_with_trend_reversal(data, index):
    # Check for shooting star pattern for the current candle
    
    if index <= 6 or index > len(data) - 6:
        return False
    
    current_candle = data.iloc[index]
    body_length = abs(current_candle['Open'] - current_candle['Close'])
    upper_shadow_length = current_candle['High'] - max(current_candle['Open'], current_candle['Close'])
    lower_shadow_length = min(current_candle['Open'], current_candle['Close']) - current_candle['Low']
    shooting_star_condition = (upper_shadow_length >= 2 * body_length) and (lower_shadow_length <= 0.5 * body_length)
    
    if not shooting_star_condition:
        return False
    
    # Check for trend reversal pattern in the previous 5 candles
    prev_candles = data.iloc[max(index - 5, 0):index]
    if len(prev_candles) < 5:
        return False
    prev_candles_trend = prev_candles['Close'].diff().mean() > 0  # Check if the trend is positive
    if prev_candles_trend:
        return False  # If the trend is positive, it's not a reversal
    
    # Check for trend reversal pattern in the next 5 candles
    next_candles = data.iloc[index + 1:min(index + 6, len(data))]
    if len(next_candles) < 5:
        return False
    next_candles_trend = next_candles['Close'].diff().mean() < 0  # Check if the trend is negative
    if next_candles_trend:
        return False  # If the trend is negative, it's not a reversal
    
    # If both conditions are met, it's a shooting star with a trend reversal
    return True

def is_shooting_star(row):
    body_length = abs(row['Open'] - row['Close'])
    upper_shadow_length = row['High'] - max(row['Open'], row['Close'])
    lower_shadow_length = min(row['Open'], row['Close']) - row['Low']
    return (upper_shadow_length >= 2 * body_length) and (lower_shadow_length <= 0.5 * body_length)

# Apply the function to each row in the DataFrame to identify shooting star candles


def prepare_data(ticker: str):
    data = get_data(
        ticker = ticker, 
        start_date = START_DATE, 
        end_date = END_DATE
    )
    # stock_data['Target'] = stock_data['Change'].shift(-5) - stock_data['Change']  # Steps 2 and 3
    # stock_data['Target'] = np.where(stock_data['Target'] >= 0, 1, 0)

    # stock_data['Low_Shadow'] = stock_data['Close'] - stock_data['Low']
    # stock_data['High_Shadow'] = stock_data['High'] - stock_data['Close']

    data['Is_Shooting_Star'] = data.apply(is_shooting_star, axis=1)
    
    # data['Shooting_Star_With_Trend_Reversal'] = [is_shooting_star_with_trend_reversal(data, index) for index in data.index]
    
    return data


# def split_data(data: pd.DataFrame):
#     target = data.Change
#     indicators = data.drop(columns=["Change"])
#     indicators_train, indicators_test, target_train, target_test =  train_test_split(indicators, target, test_size=0.2)
#     return indicators_train, indicators_test, target_train, target_test


def normalize_data(data: pd.DataFrame): 
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data)
    # scaled_data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    return data


def prepare_sequences(data: pd.DataFrame):
    dates = data.index.to_numpy()
    target = data['Direction'].to_numpy()
    indicators = data[SEQUENCE_COLUMNS].to_numpy()

    indicators = indicators[:-1]
    indicators_dates = dates[:-1]
    
    target = target[1:]
    target_dates = dates[1:]
    
    return indicators, indicators_dates, target, target_dates


def split_train_and_test_data(x, y, x_dates, y_dates):
    train = int(len(y) * .8)
    test = int(len(y) * .9)
    predict = int(len(y))
    
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


def get_cnn_data(ticker): 
    data = prepare_data(ticker)
    # data = add_lags(data)
    data =  normalize_data(data)
    indicators, indicators_dates, target, target_dates  = prepare_sequences(data)
    train, test, predict = split_train_and_test_data(indicators, target, indicators_dates, target_dates)

    return train, test, predict
