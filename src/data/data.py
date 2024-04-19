from constants import SP500_TICKER, DJI_TICKER, NASDAQ_TICKER
from data.dow_jones.dow_jones import get_dow_jones_data
from data.nasdaq.nasdaq import get_nasdaq_data
from data.sp500.sp500 import get_SP500_data 
import yfinance as yf


def get_data(
    ticker,
    start_date = "1995-10-01",
    end_date = "2024-02-01",
    interval = '1d'
):
  data = None

  if ticker == DJI_TICKER:
    data = get_dow_jones_data(start_date, end_date)
  
  elif ticker == NASDAQ_TICKER:
    data =  get_nasdaq_data(start_date, end_date)
  
  elif ticker == SP500_TICKER:
    data =  get_SP500_data(start_date, end_date)
  
  else:
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

  data["Change"] = data["Close"] - data["Open"]
  data["Direction"] = data["Change"].apply(lambda x: 1 if x > 0 else 0)

  return data
