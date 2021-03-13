from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
import glob
import cv2
import pandas as pd
import numpy as np

IMAGEWIDTH = 160
model = Sequential()

model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(IMAGEWIDTH, IMAGEWIDTH, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'test',
    target_size=(160, 160),
    batch_size=1,
    class_mode='binary',
    subset="training")


dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'train',
        target_size=(160, 160),
        batch_size=1,
        class_mode='binary')


history = model.fit_generator(
    generator,
    epochs=100,
    validation_data=validation_generator,
    validation_steps = 2700 // 30,
    steps_per_epoch = 50100 // 30
)

# Save model weights
model.save_weights("model_weight.h5")
print("Saved model to disk")

