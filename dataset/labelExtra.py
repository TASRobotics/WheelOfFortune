import cv2
import os
import os.path
import re
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('path', './extra/photos', 'where do you want to start labeling from')
flags.DEFINE_integer('start', 1, 'where do you want to start labeling from')

labels = [
    'bl', 'br', 'yl', 'yr', 'rl', 'rr', 'gl', 'gr'
]

def main(argv):   
    if not os.path.exists('./extra/labels'):
        os.mkdir('./extra/labels')
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
                img = cv2.resize(img, (int(1280 / 1.5), int(720 / 1.5)))
                
                print(str(photoN) + ": label the image using bl, br, yl, yr, rl, rr, gl, gr, or press q + enter to exit \n")
                label = ''
                while not labeled:
                    cv2.imshow('frame', img)
                    key = cv2.waitKey(0)
                    if key == 0:
                        print(label)
                        inlabel = label
                        label = ''
                        for name in labels:
                            if inlabel == name:
                                labeled = True
                                break
                        if labeled == True:
                            pass
                        elif inlabel == 'd':
                            if os.path.isfile('./extra/labels/'+filename+'.label'):
                                os.remove('./extra/labels/'+filename+'.label')
                            break
                        elif inlabel == 'q':
                            cv2.destroyAllWindows()
                            return
                        else:
                            print('That was not a label, please try again')
                            pass
                    else:
                        label += chr(key)
                    
                #write to file
                if labeled:
                    labelfile = open('./extra/labels/'+filename+'.label', "w")
                    labelfile.write(inlabel)
                    labelfile.close()

if __name__ == '__main__':
    app.run(main)