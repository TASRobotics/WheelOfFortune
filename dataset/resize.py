import cv2
import os
import re
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('path', './photos', 'path to photos')
flags.DEFINE_integer('x',1280, 'final x dimension')
flags.DEFINE_integer('y',720, 'final y dimension')

def main(argv):
    directory = os.walk(FLAGS.path+'/')
    for roots, dirs, files in directory:
        for filename in files:
            name = FLAGS.path+'/'+filename
            if re.search('.+(\.jpg)', filename):
                img = cv2.imread(name)
                resized = cv2.resize(img, (FLAGS.x, FLAGS.y), interpolation = cv2.INTER_AREA)
                cv2.imwrite(name, resized)

if __name__ == '__main__':
    app.run(main)