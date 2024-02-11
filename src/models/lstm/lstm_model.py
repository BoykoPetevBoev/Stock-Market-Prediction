
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
# from models.lstm.lstm_normalize import get_lstm_data

        
def build_model():
    INPUT_SHAPE = (5, 1)
    
    model = Sequential([
        Input(INPUT_SHAPE),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])    
    model.compile(
        loss='mse', 
        optimizer='adam', 
        metrics=['mean_absolute_error']
    )
    
    return model


def train_model(x_train, x_test, y_train, y_test):
    model = build_model()
    history = model.fit(
        x_train, 
        y_train, 
        epochs=100, 
        batch_size=32, 
        validation_data=(x_test, y_test)
    )
  
    return history
