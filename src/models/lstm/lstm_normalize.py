import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data.sp500 import get_SP500_model_data

def scale_data():
    data = get_SP500_model_data()
    scaler = MinMaxScaler()
    data['Close'] = scaler.fit_transform(np.expand_dims(data['Close'].values, axis=1)) 
    return data