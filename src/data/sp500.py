import yfinance as yf
import pandas as pd

SP500_TICKER = "^GSPC"
FILE_NAME = "sp500.csv"

def download_SP500_data(start_date, end_date):
    sp500_data = yf.download(SP500_TICKER, start=start_date, end=end_date)
    sp500_data.to_csv(FILE_NAME)  
    
def get_SP500_data(start_date, end_date): 
    sp500_data = pd.read_csv(FILE_NAME)

    sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
    
    sp500_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]

    sp500_data.set_index('Date', inplace=True)

    return sp500_data