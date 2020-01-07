from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from random import random

labels = [
    'bl', 'br', 'yl', 'yr', 'rl', 'rr', 'gl', 'gr'
]

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train_p, train_l = np.load('./dataset/data/train.npy')
test_p, test_l = np.load('./dataset/data/test.npy')

train_p = np.stack(train_p)
test_p = np.stack(test_p)

train_p = train_p / 255.0
test_p = test_p / 255.0

train_l = train_l.astype('uint8')
test_l = test_l.astype('uint8')

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(720,1280,3)),
    tf.keras.layers.Dense(1, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_p, train_l, epochs=100)

model.save('trained_model.h5')