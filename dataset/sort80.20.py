import os
import re

for root, dirs, files in os.walk("./photos/"):
    for filename in files:
        if re.search('p(.+).jpg',filename):
            if random() > 0.8:
                os.rename('./photos/'+filename, './test/photos'+filename)
                os.rename('./labels/'+filename+'.label', './test/labels'+filename+'.label')