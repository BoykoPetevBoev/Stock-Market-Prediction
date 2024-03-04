import numpy as np
import tensorflow as tf

class ModelData:
    def __init__(self, x, y, dates):
       self.x =  np.array(x)
       self.y =  np.array(y)
       self.dates =  np.array(dates)

    def get_tensors(self):
        # x = x.reshape(x.shape[0], x.shape[1], 1)
        x_tensor = tf.convert_to_tensor(self.x)
        y_tensor = tf.convert_to_tensor(self.y)
        dates_array = self.dates.copy()
        return x_tensor, y_tensor, dates_array

    def get_numpy(self):
        x_array = self.x.copy()
        y_array = self.y.copy()
        dates_array = self.dates.copy()
        return x_array, y_array, dates_array
