
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

        
def build_model():
    INPUT_SHAPE = (10, 1)
    
    model = Sequential([
        Input(INPUT_SHAPE),
        LSTM(64, return_sequences = True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
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
        epochs=100, 
        verbose=2,
        batch_size=32, 
    )
    evaluate_result = model.evaluate(
        x=x_test, 
        y=y_test, 
        batch_size=32, 
        verbose=2
    )
    return model, fit_result, evaluate_result
