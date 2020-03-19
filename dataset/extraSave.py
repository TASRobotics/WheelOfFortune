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
direct = os.walk('./extra/photos')

for dirs, roots, files in direct:
    for filename in files:
        img = cv2.imread('./extra/photos/'+filename)

        try :
            fd = os.open('./extra/labels/'+filename+'.label', os.O_RDONLY)
            readBytes = os.read(fd, 1000) 
            os.close(fd)
            label = readBytes.decode('utf8')
            
            trp.append(img)
            trl.append(classes2int(label))
        except:
            bah = 0
        
np.save('./extra/data/train.npy', (trp,trl))