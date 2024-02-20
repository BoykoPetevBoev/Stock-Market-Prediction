
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Reshape
from tensorflow.keras.optimizers import Adam

        
def build_model():
    INPUT_SHAPE = (10, 4)
    OUTPUT_SHAPE = (3, 4)
    
    model = Sequential([
        Input(INPUT_SHAPE),
        LSTM(128, return_sequences = True, input_shape=(10, 4)),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(12),
        Reshape(target_shape=OUTPUT_SHAPE)
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
