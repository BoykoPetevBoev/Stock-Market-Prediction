import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data.sp500 import get_SP500_data
import pandas_ta as ta

START_DATE = "2000-01-01"
END_DATE = "2024-02-01"
COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close']

sequence_length = 5 

def prepare_data():
    stock_data = get_SP500_data(START_DATE, END_DATE)
    stock_data = stock_data[COLUMNS]
    
    numerical_data = stock_data[COLUMNS[1:]] 
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_numerical_data = scaler.fit_transform(numerical_data)
    
    scaled_data = pd.DataFrame(scaled_numerical_data, columns=COLUMNS[1:])
    scaled_data['Date'] = stock_data['Date'].values
    scaled_data['Date'] = pd.to_datetime(scaled_data['Date'])
    
    return scaled_data

def prepare_sequences(data):
    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X_today = data['Date'].iloc[i], data['Close'].iloc[i]

        y_previous_days = data[['Open', 'High', 'Low', 'Close']].iloc[i-sequence_length:i].values.flatten()

        X.append(X_today)
        y.append(y_previous_days)   

    X = np.array(X)
    y = np.array(y)

    return X, y