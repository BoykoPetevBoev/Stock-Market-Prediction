import yfinance as yf
import pandas as pd
from constants import NASDAQ_TICKER

FILE_NAME = "./data/nasdaq/nasdaq.csv"

def download_nasdaq_data():
    nasdaq_data = yf.download(NASDAQ_TICKER, interval='1d')
    nasdaq_data.to_csv(FILE_NAME)  
    
def get_nasdaq_data(start_date, end_date): 
    nasdaq_data = pd.read_csv(FILE_NAME)
    nasdaq_data = nasdaq_data[(nasdaq_data['Date'] >= start_date) & (nasdaq_data['Date'] <= end_date)]
    return nasdaq_data
