import pandas as pd
import pandas_ta as ta
import yfinance as yf

from constants import SP500_TICKER, DJI_TICKER, NASDAQ_TICKER
from data.dow_jones.dow_jones import get_dow_jones_data
from data.nasdaq.nasdaq import get_nasdaq_data
from data.sp500.sp500 import get_SP500_data


def add_indicators(data: pd.DataFrame):
    data['MA10'] = ta.sma(data['Close'], length=10)
    data['MA20'] = ta.sma(data['Close'], length=20)
    data['MA50'] = ta.sma(data['Close'], length=50)
    data['MA100'] = ta.sma(data['Close'], length=100)

    data['RSI'] = ta.rsi(data['Close'])

    stoch_results = ta.stoch(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stochastic_K'] = stoch_results.iloc[:, 0]
    data['Stochastic_D'] = stoch_results.iloc[:, 1]

    data.dropna(inplace=True)
    return data


def get_data(
    ticker,
    start_date="1995-10-01",
    end_date="2024-02-01",
    interval='1d'
):
    data = None

    if ticker == DJI_TICKER:
        data = get_dow_jones_data(start_date, end_date)

    elif ticker == NASDAQ_TICKER:
        data = get_nasdaq_data(start_date, end_date)

    elif ticker == SP500_TICKER:
        data = get_SP500_data(start_date, end_date)

    else:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=False)

    data["Change"] = data["Close"] - data["Open"]
    data["Direction"] = data["Change"].apply(lambda x: 1 if x > 0 else 0)

    data = add_indicators(data)
    data.reset_index(inplace=True)

    return data
