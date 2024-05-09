import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import mplfinance as mpf
import matplotlib.pyplot as plt

from typing import Tuple, List
from data.data import get_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.data import Dataset
# from sklearn.preprocessing import MinMaxScaler
# from classes.model_data_class import ModelData
# from sklearn.model_selection import train_test_split


START_DATE = "1900-01-01"
END_DATE = "2024-02-01"

IMAGE_DIRECTORY = "models/cnn_v3/images/"


def prepare_data(ticker: str):
    data = get_data(
        ticker = ticker, 
        start_date = START_DATE, 
        end_date = END_DATE
    )
    data['Date']= pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    return data


def generate_images(data: pd.DataFrame):
    image_data = data[1000:2000].copy()
    num_candles = 10

    for index in range(image_data.shape[0] - num_candles):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.set_axis_off()
        end_index = index + num_candles
        formation_data = image_data[index:end_index]
        
        # mpf.plot(formation_data, type='candle', volume=False, ax=ax)
        # plt.savefig(f'{IMAGE_DIRECTORY}{image_data.Date[index]}.png', dpi=75)


def get_images():
    generator = ImageDataGenerator()
    images = generator.flow_from_directory(IMAGE_DIRECTORY)
    return images


def get_classes_and_files() -> Tuple[np.ndarray[str], np.ndarray[str]]:
    classes = os.listdir(IMAGE_DIRECTORY)
    folders = [IMAGE_DIRECTORY + class_name for class_name in classes]
    return np.array(classes), np.array(folders)


def map_classes_and_files(
        classes: np.ndarray[str], 
        folders: np.ndarray[str]
    ) -> Tuple[np.ndarray[str], np.ndarray[int]]:

    all_files = []
    all_classes = []

    for folder_name, class_name in zip(folders, classes):
        files = os.listdir(folder_name)
        all_files.extend([os.path.join(folder_name, file) for file in files])
        all_classes.extend([class_name] * len(files))

    class_mapping = {class_name: class_id for (class_id, class_name) in list(enumerate(classes))}

    all_class_ids = [class_mapping[c] for c in all_classes]

    all_files = np.array(all_files)
    all_class_ids = np.array(all_class_ids)

    return all_files, all_class_ids


def read_images(file_name: str):
    image_file = tf.io.read_file(file_name)
    image = tf.image.decode_png(image_file)
    image_scaled = tf.cast(image, float) / 255.0
    image_resized = tf.image.resize(image_scaled, (256, 256))
    return image_resized


def get_images_dataset(
        all_files: np.ndarray[str], 
        class_ids: np.ndarray[int]
    ) -> Tuple[np.ndarray, np.ndarray[int]]:
    # dataset = Dataset \
    #     .from_tensor_slices((all_files, all_class_ids)) \
    #     .shuffle(len(all_files)) \
    #     .map(read_images) \
    #     .batch(1) \
    #     .repeat()
    images = [read_images(file_path) for file_path in all_files]
    images = np.array(images)
    return images, class_ids


def shuffle_images_and_labels(
        images: np.ndarray, 
        labels: np.ndarray[int]
    ) -> Tuple[np.ndarray, np.ndarray[int]]:
    combined = list(zip(images, labels))
    random.shuffle(combined)
    shuffled_images, shuffled_labels = zip(*combined)
    return np.array(shuffled_images), np.array(shuffled_labels)


def split_train_and_test_data(
        x: np.ndarray,  
        y: np.ndarray[int]
    ):
    train = int(len(y) * .8)
    test = int(len(y) * .9)
    predict = int(len(y))
    
    x_train, y_train = x[0:train], y[0:train]
    x_test, y_test = x[train:test], y[train:test]
    x_predict, y_predict = x[test:predict], y[test:predict]
    
    return {
        'x': x[0:train],
        'y': y[0:train],
    }, {
        'x': x_test,
        'y': y_test,
    }, {
        'x': x_predict,
        'y': y_predict,
    }


def get_cnn_data(ticker: str): 
    classes, folders = get_classes_and_files()
    all_files, all_class_ids = map_classes_and_files(classes, folders)
    images, labels = get_images_dataset(all_files, all_class_ids)
    images, labels = shuffle_images_and_labels(images, labels)
    train_dataset, test_dataset, predict_dataset = split_train_and_test_data(images, labels)

    return train_dataset, test_dataset, predict_dataset
