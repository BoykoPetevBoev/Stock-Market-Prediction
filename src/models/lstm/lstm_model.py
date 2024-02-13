
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
# from models.lstm.lstm_normalize import get_lstm_data

        
def build_model():
    INPUT_SHAPE = (10, 1)
    
    model = Sequential([
        Input(INPUT_SHAPE),
        LSTM(units = 50, return_sequences = True),
        Dropout(0.2),

        LSTM(units = 50, return_sequences = True),
        Dropout(0.2),

        LSTM(units = 50, return_sequences = True),
        Dropout(0.2),

        LSTM(units = 50),
        Dropout(0.2),

        Dense(units = 1),
        
        
        # LSTM(32),
        # LSTM(64),
        # Dense(16, activation='relu'),
        # Dropout(0.2),
        # Dense(16, activation='relu'),
        # Dropout(0.2),
        # Dense(1)
    ])    
    model.compile(
        optimizer=Adam(0.001), 
        loss='mse', 
        metrics=['mean_absolute_error']
    )
    return model


def train_model(x_train, x_test, y_train, y_test):
    model = build_model()
    fit_result = model.fit(
        x=x_train, 
        y=y_train, 
        epochs=10, 
        # verbose=2,
        # batch_size=32, 
    )
    evaluate_result = model.evaluate(
        x=x_test, 
        y=y_test, 
        # batch_size=32, 
        # verbose=2
    )
  
    return model, fit_result, evaluate_result
