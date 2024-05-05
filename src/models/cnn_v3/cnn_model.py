import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


CNN_V2_MODEL_DIRECTORY = './models/cnn_v3/cnn_model_v3.keras'        
CNN_V2_LOG_DIRECTORY = 'logs/cnn_v3'
INPUT_SHAPE = (256, 256, 4)


def build_model():
    model = Sequential([
        Input((256, 256, 4)),
        Conv2D(filters=32, kernel_size=(3, 4), activation='relu'),
        Conv2D(filters=16, kernel_size=(3, 4), activation='relu'),
        Conv2D(filters=8, kernel_size=(3, 4), activation='relu'),
        Flatten(),
        Dense(units=32, activation='relu'),
        Dense(units=5, activation='softmax')
    ])
    model.compile(
        # optimizer=Adam(0.001), 
        loss='sparse_categorical_crossentropy', 
        metrics=[
            # 'mean_absolute_error',
            'accuracy'
        ]
    )
    return model


def train_model(
    train_dataset: tf.Tensor, 
    test_dataset: tf.Tensor
):
    model = build_model()
    tensorboard_callback = TensorBoard(log_dir=CNN_V2_LOG_DIRECTORY)
    
    fit_result = model.fit(
        train_dataset,
        epochs=5, 
        steps_per_epoch=50,
        callbacks=[tensorboard_callback]
    )
    # evaluate_result = model.evaluate(
    #     test_dataset, 
    # )
    return model, fit_result, fit_result


def save_cnn_model(model):
    model.save(CNN_V2_MODEL_DIRECTORY)

    
def load_cnn_model():
    model = load_model(CNN_V2_MODEL_DIRECTORY)
    return model
    
