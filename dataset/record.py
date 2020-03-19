import numpy as np
import cv2

cap = cv2.VideoCapture(1)
i = 0
photoN = 2200
save_path = "./extra/photos"

while True:
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    i += 1

    if i == 10:
        cv2.imwrite(save_path+'/'+"%s.jpg" % (photoN), frame)
        photoN += 1
        i = 0