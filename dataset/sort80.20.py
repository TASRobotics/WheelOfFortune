import os
import re
from random import random

for root, dirs, files in os.walk("./photos/"):
    for filename in files:
        if re.search('(.+).jpg',filename):
            print('a')
            if random() > 0.8:
                os.rename('./photos/'+filename, './test/photos/'+filename)
                os.rename('./labels/'+filename+'.label', './test/labels/'+filename+'.label')