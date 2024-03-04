import matplotlib.pyplot as plt
import models.lstm_v1.lstm_model as lstm_v1
import models.lstm_v1.lstm_normalize as data_v1
import models.lstm_v2.lstm_model as lstm_v2 
import models.lstm_v2.lstm_normalize as data_v2

def predict_price_lstm_v1_model():
    train, test, predict = data_v1.get_lstm_data()
    x_predict, y_predict, dates_predict = predict.get_tensors()
    
    lstm_v1_model = lstm_v1.load_lstm_model()
    predictions = lstm_v1_model.predict(x_predict)
    return predictions 
    
    
