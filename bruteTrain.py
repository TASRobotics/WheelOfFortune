from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from random import random
from absl import app
from absl import flags

labels = []
train_p = []
train_l = []
test_p = []
test_l = []
accHist = []

#prep data
def prep():
    global labels
    global train_p
    global train_l
    global test_p
    global test_l

    labels = [
        'bl', 'br', 'yl', 'yr', 'rl', 'rr', 'gl', 'gr'
    ]
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    
    train_pic, train_lab = np.load('./dataset/data/train.npy')
    test_pic, test_lab = np.load('./dataset/data/test.npy')

    train_pic = np.stack(train_pic)
    test_pic = np.stack(test_pic)

    train_p = train_pic / 255.0
    test_p = test_pic / 255.0

    train_l = train_lab.astype('uint8')
    test_l = test_lab.astype('uint8')
    print('succsessfully loaded datasets')

#model structure
def runModel(layerAr):
    global accHist
    global train_l
    global train_p
    global test_l
    global test_p

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',input_shape=(360,640,3)),
            tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
        ])


        print(test_l)
        for nodeN in layerAr:
            if not nodeN == 0: 
                print(nodeN)
                #model.add(tf.keras.layers.Dense(nodeN * 10, activation='relu'))
            else:
                break

        model.add(tf.keras.layers.Dense(8, activation='softmax'))

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        print('sda')

        model.fit(train_p, train_l, batch_size=4, epochs=2)
        
        test_loss, test_acc = model.evaluate(test_p,  test_l, verbose=2)
        accHist = np.append(test_acc, accHist)
#

layers = 0
nodes = 0
hidden = []
#layer
def newLayers(more, perLayer):
    global layers
    global hidden
    # stuff you run before the loop
    next = more - 1
    #
    for rep in range(perLayer):
        # actual repeated function run before every loop
        #
        if next == 0:
            #function run in last loop
            #runModel(hidden)
            print(hidden)
            #
        else:
            newLayers(next, perLayer)
        
        # actual repeated function run before every loop
        hidden[next] += 1
        #
    
    hidden[next] =  1 


#brute forceing
def brute():
    global layers
    global nodes
    global hidden
    layers = 5
    nodes = 15
    for i in range(layers):
        hidden.append(0)
    newLayers(layers, nodes)
    print(hidden)

def main(argv):
    global accHist
    accHist = np.asarray(accHist)
    accHist = accHist.astype('float')
    prep()
    #brute()
    test = [5]
    runModel(test)

    np.save('accuracy.npy', accHist)
    print('accuracy list:')
    print(accHist)
    print('highest accuracy:')
    print(accHist[np.argmax(accHist)])
    print('location: ')
    print(np.argmax(accHist))
if __name__ == '__main__':
    app.run(main)