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

        try :
            fd = os.open('./labels/'+filename+'.label', os.O_RDONLY)
            readBytes = os.read(fd, 1000) 
            os.close(fd)
            label = readBytes.decode('utf8')
            
            trp.append(img)
            trl.append(classes2int(label))
        except:
            bah = 0

direct = os.walk('./test/photos')

for dirs, roots, files in direct:
    for filename in files:
        img = cv2.imread('./test/photos/'+filename)

        try :
            fd = os.open('./test/labels/'+filename+'.label', os.O_RDONLY)
            readBytes = os.read(fd, 1000) 
            os.close(fd)
            label = readBytes.decode('utf8')
            tep.append(img)
            tel.append(classes2int(label))
        except: 
            bah = 0
        
np.save('./data/train.npy', (trp,trl))
np.save('./data/test.npy', (tep,tel))