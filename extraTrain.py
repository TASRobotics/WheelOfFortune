from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
from random import random

BATCH_SIZE = 256
EPOCHS = 30

labels = [
    'bl', 'br', 'yl', 'yr', 'rl', 'rr', 'gl', 'gr'
]

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train_p, train_l = np.load('./dataset/extra/data/train.npy')

train_p = np.stack(train_p)

train_p = train_p / 255.0

train_l = train_l.astype('uint8')


model = load_model('working_model_1.1.1.h5')

model.fit(train_p, train_l, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True)

model.save('trained_model.h5')