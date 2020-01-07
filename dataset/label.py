import cv2
import os
import os.path
import re
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('path', './photos', 'where do you want to start labeling from')
flags.DEFINE_integer('start', 1, 'where do you want to start labeling from')

labels = [
    'bl', 'br', 'yl', 'yr', 'rl', 'rr', 'gl', 'gr'
]

def main(argv):   
    if not os.path.exists('./labels'):
        os.mkdir('./labels')
    directory = os.walk(FLAGS.path+'/')
    photoN = 0
    for roots, dirs, files in directory:
        for filename in files:
            if re.search('.+(\.jpg)', filename):
                photoN += 1
                if photoN < FLAGS.start:
                    continue
                labeled = False
                #read image, and input label
                img = cv2.imread(FLAGS.path+'/'+filename)
                img = cv2.resize(img, (int(1280/1.5), int(720 / 1.5)))
                cv2.imshow('frame', img)
                cv2.waitKey(1)
                
                while not labeled:
                    label = input(str(photoN) + ": label the image using bl, br, yl, yr, rl, rr, gl, gr, or press ctrl + c to exit \n")
                    for name in labels:
                        if label == name:
                            labeled = True
                            break
                        elif name == 'gr':
                            print('That was not a label, please try again')                    
                    
                #write to file
                labelfile = open('./labels/'+filename+'.label', "w")
                labelfile.write(label)
                labelfile.close()

if __name__ == '__main__':
    app.run(main)