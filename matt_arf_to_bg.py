import sys
import os
import subprocess
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage, signal
import math
import datetime
import copy
import transforms3d as tf3d
import time
import random
import matplotlib.pyplot as plt
from scipy.ndimage import shift


if __name__ == '__main__':
    img_path = sys.argv[1]


    dep_img = cv2.imread(img_path, -1)