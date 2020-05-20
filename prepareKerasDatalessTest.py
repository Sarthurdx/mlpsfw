import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.utils import shuffle
import os

def splitData():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x,y = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
    x = (x-127.5) / 128
    y = keras.utils.to_categorical(y)
    seed = 24601
    x,y = shuffle(x,y, random_state=seed)
    section = int(x.shape[0] / 2)
    x_in, y_in = x[:section], y[:section]
    x_out, y_out = x[section:], y[section:]
    np.savez('cifarIN.npz', x_in, y_in)
    np.savez('cifarOUT.npz', x_out, y_out)


def makeCifarModel():
  model = Sequential()

  model.add(Conv2D(32, (3, 3), padding='same',
                  input_shape=x_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
  model.fit(x_in, y_in, epochs=500, batch_size=64)
  model.save('kerasCifar10model_500_64')
  return model
