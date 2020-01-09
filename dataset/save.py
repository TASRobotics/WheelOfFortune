import cv2
import numpy as np
import os

labels = [
    'bl', 'br', 'yl', 'yr', 'rl', 'rr', 'gl', 'gr'
]

def classes2int(label):
    return labels.index(label)

trp=[]
trl=[]
tep=[]
tel=[]
direct = os.walk('./photos')

for dirs, roots, files in direct:
    for filename in files:
        img = cv2.imread('./photos/'+filename)
        trp.append(img)

        fd = os.open('./labels/'+filename+'.label', os.O_RDONLY)
        readBytes = os.read(fd, 2) 
        os.close(fd)
        label = readBytes.decode('utf8')
        trl.append(classes2int(label))

direct = os.walk('./test/photos')

for dirs, roots, files in direct:
    for filename in files:
        img = cv2.imread('./test/photos/'+filename)
        tep.append(img)

        fd = os.open('./test/labels/'+filename+'.label', os.O_RDONLY)
        readBytes = os.read(fd, 1000) 
        os.close(fd)
        label = readBytes.decode('utf8')
        tel.append(classes2int(label))

np.save('./data/train.npy', (trp,trl))
np.save('./data/test.npy', (tep,tel))