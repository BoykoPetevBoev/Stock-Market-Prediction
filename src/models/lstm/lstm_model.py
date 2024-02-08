
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from models.lstm.lstm_normalize import prepare_data, prepare_sequences

def get_data():
    scaled_data = prepare_data()
    X, y = prepare_sequences(scaled_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    return X_train, X_test, y_train, y_test

    
def build_model():
    INPUT_SHAPE = (1, 1)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=INPUT_SHAPE),
        Dense(20)
    ])    
    model.compile(optimizer='adam', loss='mse')
    
    return model


def train_model():
    X_train, X_test, y_train, y_test = get_data()
  
    model = build_model()
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
  
    return history
