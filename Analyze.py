import os
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import measure
import sys

FILEPATH = ''

if len(sys.argv) != 1:
    FILEPATH =


#metadata = '../input/deepfake-detection-challenge/train_sample_videos/metadata.json'

# load the filenames for train videos
train_fns = sorted(glob.glob(TRAIN_PATH + '*.mp4'))

# load the filenames for test videos
#test_fns = sorted(glob.glob(TEST_PATH + '*.mp4'))

print('There are {} samples in the train set.'.format(len(train_fns)))
#print('There are {} samples in the test set.'.format(len(test_fns)))