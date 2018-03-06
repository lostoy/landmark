import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch

import h5py
import bisect
import os
import cv2
from PIL import Image
def img_loader(img_path, config):
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



if __name__ == '__main__':
    pass