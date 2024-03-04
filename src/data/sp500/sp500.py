import yfinance as yf
import pandas as pd

SP500_TICKER = "^GSPC"
FILE_NAME = "./data/sp500/sp500.csv"

def download_SP500_data(start_date, end_date):
    sp500_data = yf.download(SP500_TICKER, start=start_date, end=end_date, interval='1d')
    sp500_data.to_csv(FILE_NAME)  
    
def get_SP500_data(start_date, end_date): 
    sp500_data = pd.read_csv(FILE_NAME)
    sp500_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]
    sp500_data["Change"] = sp500_data["Close"] - sp500_data["Open"]
    return sp500_data
