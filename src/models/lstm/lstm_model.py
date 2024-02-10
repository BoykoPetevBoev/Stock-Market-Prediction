
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from models.lstm.lstm_normalize import get_lstm_data

        
def build_model():
    INPUT_SHAPE = (5, 1)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=INPUT_SHAPE),
        Dense(20)
    ])    
    model.compile(optimizer='adam', loss='mse')
    
    return model


def train_model():
    X_train, X_test, y_train, y_test = get_lstm_data()
  
    model = build_model()
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
  
    return history
