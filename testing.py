import tensorflow as tf
import os

model = tf.keras.models.load_model('models/vgg-3c1f-1623404835')

data_path = 'data/'
data_dir_list = os.listdir(data_path)
