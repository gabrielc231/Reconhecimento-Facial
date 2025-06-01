import tensorflow as tf
from keras import models
import keras.utils
import os

folder_path = "models/"
file_path = "model6.h5"
path = folder_path+file_path
print(path)
model = models.load_model(path)
print(model)
keras.utils.plot_model(model)
