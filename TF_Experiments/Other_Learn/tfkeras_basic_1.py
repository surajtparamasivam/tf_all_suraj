import tensorflow as tf
from tensorflow import keras

import numpy as np
from matplotlib import pyplot as plt

dataset1=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = dataset1.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']