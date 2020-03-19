# load and evaluate a saved model
from numpy import loadtxt
import numpy as np
from tensorflow.keras.models import load_model
import cv2

labels = [
    'bl', 'br', 'yl', 'yr', 'rl', 'rr', 'gl', 'gr'
]

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
# load model
model = load_model('trained_model.h5')
#model = load_model('working_model_1.1.0.h5')

# summarize model.
model.summary()
# evaluate the model
#test_p, test_l = np.load('./dataset/data/test.npy')
#
#test_p = np.stack(test_p)
#
#test_p = test_p / 255.0
#test_l = test_l.astype('uint8')

#pre = model.predict(test_p)
#print(preVideoCapture(1)
#index = 0
#correct = 0
#for owo in pre:
#    if np.argmax(owo) == test_l[index]:
#        correct += 1
#    index += 1
#
#correct = correct / index
#print("accuracy: " + str(correct))
cap = cv2.VideoCapture(0)

success, frame = cap.read()
while success:
    #frame = cv2.resize(frame, (640, 360), interpolation = cv2.INTER_AREA)
    frame = cv2.resize(frame, (400, 225), interpolation = cv2.INTER_AREA)

    img = []
    img.append(frame)
    img = np.stack(img)
    img = img / 255

    res = model.predict(img)

    # print most likely result
    #print(str(labels[np.argmax(res)]))
    
    result = np.zeros((225, 400, 3), 'uint8')
    cv2.rectangle(result, (0, 225), (50, 225 - int(res[0][0]*225)), (255,0,0), -1)
    cv2.rectangle(result, (50, 225), (100, 225 - int(res[0][1]*225)), (255,0,0),-1)
    cv2.rectangle(result, (100, 225), (150, 225 - int(res[0][2]*225)), (0,200,200), -1)
    cv2.rectangle(result, (150, 225), (200, 225 - int(res[0][3]*225)), (0,200,200), -1)
    cv2.rectangle(result, (200, 225), (250, 225 - int(res[0][4]*225)), (0,0,255), -1)
    cv2.rectangle(result, (250, 225), (300, 225 - int(res[0][5]*225)), (0,0,255), -1)
    cv2.rectangle(result, (300, 225), (350, 225 - int(res[0][6]*225)), (0,255,0), -1)
    cv2.rectangle(result, (350, 225), (400, 225 - int(res[0][7]*225)), (0,255,0), -1)
    cv2.imshow('result', result)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    success, frame = cap.read()
