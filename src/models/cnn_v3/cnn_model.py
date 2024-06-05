import tensorflow as tf
import pandas as pd

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input


CNN_V2_MODEL_DIRECTORY = './models/cnn_v3/cnn_model_v3.keras'        
CNN_V2_LOG_DIRECTORY = 'logs/cnn_v3'
INPUT_SHAPE = (256, 256, 4)


def build_model():
    model = Sequential([
        Input(INPUT_SHAPE),
        Conv2D(filters=32, kernel_size=(3, 4), activation='relu'),
        Conv2D(filters=16, kernel_size=(3, 4), activation='relu'),
        Conv2D(filters=8, kernel_size=(3, 4), activation='relu'),
        Flatten(),
        Dense(units=32, activation='relu'),
        Dense(units=5, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            # tf.keras.metrics.Precision(),
            # tf.keras.metrics.Recall()
        ]
    )
    return model


def train_model(
    x_train,
    x_test,
    y_train,
    y_test,
):
    model = build_model()
    tensorboard_callback = TensorBoard(log_dir=CNN_V2_LOG_DIRECTORY)
    
    fit_result = model.fit(
        x=x_train, 
        y=y_train,
        epochs=8, 
        steps_per_epoch=100,
        callbacks=[tensorboard_callback]
    )
    evaluate_result = model.evaluate(
        x=x_test, 
        y=y_test,
    )
    return model, fit_result, evaluate_result


def save_cnn_model(model):
    model.save(CNN_V2_MODEL_DIRECTORY)

    
def load_cnn_model():
    model = load_model(CNN_V2_MODEL_DIRECTORY)
    return model

