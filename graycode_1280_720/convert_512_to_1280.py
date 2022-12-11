import numpy as np
import cv2
import os

source = '/home/stefan/transparent_target_rendering/graycode_512_512'
target = '/home/stefan/transparent_target_rendering/graycode_256_256'

images = os.listdir(source)

for name in images:
    load = os.path.join(source, name)
    img = cv2.imread(load)
    img = cv2.resize(img, (256, 256))
    save = os.path.join(target, name)
    cv2.imwrite(save, img)


