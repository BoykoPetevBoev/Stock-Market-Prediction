import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


CNN_V1_MODEL_DIRECTORY = './models/cnn_v1/cnn_model_v1'        
CNN_V1_LOG_DIRECTORY = 'logs/cnn_v1'
INPUT_SHAPE = (10, 1)


def build_model():
    model = Sequential([
        Conv1D(
            filters=64, 
            kernel_size=(3,), 
            activation='relu', 
            input_shape=INPUT_SHAPE
        ),
        MaxPooling1D(
            pool_size=(2,)
        ),
        Conv1D(
            filters=64, 
            kernel_size=(3,), 
            activation='relu'
        ),
        MaxPooling1D(
            pool_size=(2,)
        ),
        Flatten(),
        Dense(
            units=1, 
            activation='linear'
        )
    ])
    model.compile(
        optimizer=Adam(0.001), 
        loss='mse', 
        metrics=[
            'mean_absolute_error',
            'accuracy'
        ]
    )
    return model


def train_model(
    x_train: tf.Tensor, 
    x_test: tf.Tensor, 
    y_train: tf.Tensor, 
    y_test: tf.Tensor
):
    model = build_model()
    tensorboard_callback = TensorBoard(log_dir=CNN_V1_LOG_DIRECTORY)
    
    fit_result = model.fit(
        x=x_train, 
        y=y_train, 
        epochs=200, 
        verbose=2,
        batch_size=32,
        callbacks=[tensorboard_callback]
    )
    evaluate_result = model.evaluate(
        x=x_test, 
        y=y_test, 
        batch_size=32, 
        verbose=2
    )
    return model, fit_result, evaluate_result


def save_cnn_model(model):
    model.save(CNN_V1_MODEL_DIRECTORY)

    
def load_cnn_model():
    model = load_model(CNN_V1_MODEL_DIRECTORY)
    return model
    
