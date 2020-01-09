# load and evaluate a saved model
from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import load_model



np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# load model
model = load_model('trained_model.h5')
# summarize model.
model.summary()
# evaluate the model
test_p, test_l = np.load('./dataset/data/test.npy')

test_p = np.stack(test_p)

test_p = test_p / 255.0
test_l = test_l.astype('uint8')

pre = model.predict(test_p)