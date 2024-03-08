import yfinance as yf
import pandas as pd
from constants import DJI_TICKER

FILE_NAME = "./data/dow_jones/dow_jones.csv"

def download_dow_jones_data():
    dow_jones_data = yf.download(DJI_TICKER, interval='1d')
    dow_jones_data.to_csv(FILE_NAME)  
    
def get_dow_jones_data(start_date, end_date): 
    dow_jones_data = pd.read_csv(FILE_NAME)
    dow_jones_data = dow_jones_data[(dow_jones_data['Date'] >= start_date) & (dow_jones_data['Date'] <= end_date)]
    dow_jones_data["Change"] = dow_jones_data["Close"] - dow_jones_data["Open"]
    return dow_jones_data
