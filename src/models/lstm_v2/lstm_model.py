
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Reshape, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model


LSTM_V2_MODEL_DIRECTORY = './models/lstm_v2/lstm_model_v2'        
LSTM_V2_LOG_DIRECTORY = 'logs/lstm_v2'      
INPUT_SHAPE = (15, 3)
OUTPUT_SHAPE = (2, 3)

    
def build_model():
    model = Sequential([
        # Input(INPUT_SHAPE),
        LSTM(90, input_shape=INPUT_SHAPE),
        # LSTM(90),
        # Dense(90, activation='relu'),
        Dense(45, activation='relu'),
        Dense(45, activation='relu'),
        Dense(45, activation='relu'),
        # Dense(5),
        Dense(15 * 3),
        Dense(2 * 3),
        Reshape(OUTPUT_SHAPE)
    ])    
    model.compile(
        optimizer=Adam(0.001), 
        loss='mse', 
        metrics=['mean_absolute_error']
    )
    return model


def train_model(x_train, x_test, y_train, y_test):
    model = build_model()
    tensorboard_callback = TensorBoard(log_dir=LSTM_V2_LOG_DIRECTORY)
    
    fit_result = model.fit(
        x=x_train, 
        y=y_train, 
        epochs=200,
        verbose=2,
        # batch_size=32,
        callbacks=[tensorboard_callback]
    )
    evaluate_result = model.evaluate(
        x=x_test, 
        y=y_test, 
        # batch_size=32, 
        verbose=2
    )
    return model, fit_result, evaluate_result


def save_lstm_model(model):
    model.save(LSTM_V2_MODEL_DIRECTORY)

    
def load_lstm_model():
    model = load_model(LSTM_V2_MODEL_DIRECTORY)
    return model
    
