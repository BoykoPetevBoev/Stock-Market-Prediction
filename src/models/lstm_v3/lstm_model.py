import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model


LSTM_V2_MODEL_DIRECTORY = './models/lstm_v3/lstm_model_v3.keras'        
LSTM_V2_LOG_DIRECTORY = 'logs/lstm_v3'      
INPUT_SHAPE = (10, 3)
# OUTPUT_SHAPE = (1)

    
def build_model():
    model = Sequential([
        Input(INPUT_SHAPE),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation="softmax")
    ])    
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    return model


def train_model(x_train, x_test, y_train, y_test):
    model = build_model()
    tensorboard_callback = TensorBoard(log_dir=LSTM_V2_LOG_DIRECTORY)
    
    fit_result = model.fit(
        x=x_train, 
        y=y_train, 
        epochs=100,
        callbacks=[tensorboard_callback]
    )
    evaluate_result = model.evaluate(
        x=x_test, 
        y=y_test, 
    )
    return model, fit_result, evaluate_result


def save_lstm_model(model):
    model.save(LSTM_V2_MODEL_DIRECTORY)

    
def load_lstm_model():
    model = load_model(LSTM_V2_MODEL_DIRECTORY)
    return model
    
