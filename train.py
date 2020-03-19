from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
from random import random

BATCH_SIZE = 350
EPOCHS = 80

labels = [
    'bl', 'br', 'yl', 'yr', 'rl', 'rr', 'gl', 'gr'
]

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train_p, train_l = np.load('./dataset/data/train.npy')

train_p = np.stack(train_p)

train_p = train_p / 255.0

train_l = train_l.astype('uint8')

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():

    #model = tf.keras.Sequential([
    #    tf.keras.layers.Conv2D(16, (6,6), activation='relu',input_shape=(360,640,3)),
    #    tf.keras.layers.Conv2D(8, (3,3), activation='relu',input_shape=(360,640,3)),
    #    tf.keras.layers.Conv2D(4, (2,2), activation='relu',input_shape=(360,640,3)),
    #    tf.keras.layers.Flatten(),
    #])
#
    #model.add(tf.keras.layers.Dense(128, activation='relu'))
    #model.add(tf.keras.layers.Dense(32, activation='relu'))
    #model.add(tf.keras.layers.Dense(16, activation='relu'))
    #model.add(tf.keras.layers.Dense(8, activation='softmax'))

    model = tf.keras.Sequential([
        #input 224x224 rgb image
        tf.keras.layers.Conv2D(96, (11, 11), strides = 4, input_shape=[225, 400, 3], activation='relu'),
        # newSize = roundUp( (size - (filterSize - 1) )/ strides )
        # outputs image of (54, 54, 96)

        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),

        tf.keras.layers.Conv2D(256, (5, 5), activation='relu'),

        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),

        tf.keras.layers.Conv2D(384, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(384, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu'),

        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(8, activation='softmax')
    ])

    adam = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(train_p, train_l, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True)

    model.save('trained_model.h5')