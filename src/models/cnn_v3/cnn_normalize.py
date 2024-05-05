import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data.data import get_data
from sklearn.preprocessing import MinMaxScaler
from classes.model_data_class import ModelData
from sklearn.model_selection import train_test_split


START_DATE = "1900-01-01"
END_DATE = "2024-02-01"
IMAGE_DIRECTORY = "models/cnn_v3/data/"



SEQUENCE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Direction']
OUTPUT_COLUMNS = ['Change']

SEQUENCE_LENGTH = 1
OUTPUT_LENGTH = 1


def prepare_data(ticker: str):
    data = get_data(
        ticker = ticker, 
        start_date = START_DATE, 
        end_date = END_DATE
    )
    data['Date'] = data.index.strftime('%Y-%m-%d')
    data = data.reset_index(drop=True)
    return data


def generateImages(data):
    image_data = data[0:1000].copy()

    num_candles = 10

    for index in range(image_data.shape[0] - num_candles):

        fig, ax = plt.subplots(figsize=(2, 2))
        end_index = index + num_candles
        formation_data = image_data[index:end_index]
        ax.set_axis_off()

        # mpf.plot(formation_data, type='candle', volume=False, ax=ax)
        # plt.savefig(f'models/cnn_v3/data/{shooting_star_formations.Date[index]}.png', dpi=75)  # Adjust dpi for desired resolution


def getImages():
    generator = ImageDataGenerator()
    images = generator.flow_from_directory(IMAGE_DIRECTORY)
    return images


def getClassesAndFiles():
    classes = os.listdir(IMAGE_DIRECTORY)
    folders = [IMAGE_DIRECTORY + class_name for class_name in classes]
    return classes, folders


def mapAllClassesAndImages(classes, folders):
    all_files = []
    all_classes = []

    for folder_name, class_name in zip(folders, classes):
        files = os.listdir(folder_name)
        all_files.extend([os.path.join(folder_name, file) for file in files])
        all_classes.extend([class_name] * len(files))

    class_mapping = {class_name: class_id for (class_id, class_name) in list(enumerate(classes))}

    all_class_ids = [class_mapping[c] for c in all_classes]
    return all_files, all_class_ids



def read_images(file_name, class_name):
    image_file = tf.io.read_file(file_name)
    image = tf.image.decode_png(image_file)
    image_scaled = tf.cast(image, float) / 255.0
    image_resized = tf.image.resize(image_scaled, (256, 256))
    return (image_resized, class_name)


def getImagesDataset(all_files, all_class_ids):
    dataset = Dataset \
        .from_tensor_slices((all_files, all_class_ids)) \
        .shuffle(len(all_files)) \
        .map(read_images) \
        .batch(1) \
        # .repeat()

    return dataset


def split_train_and_test_data(dataset):
    length = tf.data.experimental.cardinality(dataset).numpy()

    train_size = int(0.7 * length)
    test_size = int(0.2 * length)
    predict_size = length - train_size - test_size

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    predict_dataset = dataset.skip(train_size + test_size)

    return train_dataset, test_dataset, predict_dataset



def prepare_sequences(data: pd.DataFrame):
    dates = data.index.to_numpy()
    target = data['Direction'].to_numpy()
    indicators = data[SEQUENCE_COLUMNS].to_numpy()

    indicators = indicators[:-1]
    indicators_dates = dates[:-1]
    
    target = target[1:]
    target_dates = dates[1:]
    
    return indicators, indicators_dates, target, target_dates


def get_cnn_data(ticker): 
    classes, folders = getClassesAndFiles()
    all_files, all_class_ids = mapAllClassesAndImages(classes, folders)
    dataset = getImagesDataset(all_files, all_class_ids)
    train_dataset, test_dataset, predict_dataset = split_train_and_test_data(dataset)

    return train_dataset, test_dataset, predict_dataset
