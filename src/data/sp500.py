import yfinance as yf
import pandas as pd

SP500_TICKER = "^GSPC"
FILE_NAME = "./data/sp500.csv"

def download_SP500_data(start_date, end_date):
    sp500_data = yf.download(SP500_TICKER, start=start_date, end=end_date, interval='1d')
    sp500_data.to_csv(FILE_NAME)  
    
def get_SP500_data(start_date, end_date): 
    sp500_data = pd.read_csv(FILE_NAME)

    sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
    
    sp500_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]

    sp500_data.set_index('Date', inplace=True)

    return sp500_data

def get_SP500_model_data():
    start_date = "2000-01-01"
    end_date = "2024-02-01"
    
    data = get_SP500_data(start_date, end_date)
    # remove columns which our neural network will not use
    # selected_columns = ['Date', 'Close']
    # filtered_data = data[selected_columns]
    # filtered_data = filtered_data.reset_index()
    
    # remove columns which our neural network will not use
    filtered_data = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

    # create the column 'date' based on index column
    # filtered_data['date'] = filtered_data.index
    # create the column 'date' based on index column
    # filtered_data.set_index('Date', inplace=True)
    
    return filtered_data