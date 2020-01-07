import cv2
import os
import re
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('videos_path', './vids', 'path to videos')
flags.DEFINE_string('save_path', './photos', 'path to saving the frames')
flags.DEFINE_string('video_extension', 'mp4', 'the extension the videos use (ex mp4)')
flags.DEFINE_integer('fpi', 5, 'frames per image: how many frames you read for every frame saved')

def main(argv):
    directory = os.walk(FLAGS.videos_path+'/')
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    for roots, dirs, files in directory:
        photoN = 0
        for filename in files:
            if re.search('.+(\.%s)' % FLAGS.video_extension, filename):
                vidcap = cv2.VideoCapture(FLAGS.videos_path+'/'+filename)
                success,image = vidcap.read()
                fcount = 0
                while success:
                    if fcount % FLAGS.fpi == 0:
                        cv2.imwrite(FLAGS.save_path+'/'+"%s.jpg" % (photoN), image)     # save frame as JPEG file
                        photoN += 1  
                    success,image = vidcap.read()
                    fcount += 1

if __name__ == '__main__':
  app.run(main)