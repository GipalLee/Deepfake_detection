from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
import glob
import cv2
import pandas as pd
import numpy as np
import keras
from classifiers import *


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGEWIDTH = 160
model = Meso4()
model.load('MesoNet-master/weights/Meso4_DF.h5')


dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'f2f_mesonet_validation',
        target_size=(160, 160),
        batch_size=1,
        class_mode='binary',
        subset='training')

# 3 - Predict
X, y = generator.next()
print('Predicted :', model.predict(X), '\nReal class :', y)
