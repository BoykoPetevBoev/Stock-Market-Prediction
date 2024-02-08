import yfinance as yf
import pandas as pd
from utils.date import string_to_datetime

SP500_TICKER = "^GSPC"
FILE_NAME = "./data/sp500.csv"

def download_SP500_data(start_date, end_date):
    sp500_data = yf.download(SP500_TICKER, start=start_date, end=end_date, interval='1d')
    sp500_data.to_csv(FILE_NAME)  
    
def get_SP500_data(start_date, end_date): 
    sp500_data = pd.read_csv(FILE_NAME)

    sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
    
    sp500_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]

    # sp500_data.set_index('Date', inplace=True)

    return sp500_data

def get_SP500_model_data():
    start_date = "2000-01-01"
    end_date = "2024-02-01"
    columns = ['Date', 'Close']
    
    data = get_SP500_data(start_date, end_date)
    filtered_data = data[columns]
    # filtered_data['Date'] = filtered_data['Date'].apply(string_to_datetime)
    filtered_data.index = filtered_data.pop('Date')

    return filtered_data