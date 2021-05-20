import argparse
import glob
import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import skimage.measure
from skimage import measure
from facenet_pytorch import MTCNN
from PIL import Image


# load the filenames for train videos
TRAIN_PATH = './dataset/faceswap/'
train_fns = sorted(glob.glob(TRAIN_PATH + '*.mp4'))
print('There are {} samples in the train set.'.format(len(train_fns)))


# 비디오에서 프레임따기
def get_frames(filename):
    '''
    Get all frames from the video
    INPUT:
        filename - video filename
    OUTPUT:
        frames - the array of video frames
    '''
    frames = []

    cap = cv2.VideoCapture(filename)

    # while(cap.isOpened()):
    while len(frames) < 241:
        ret, frame = cap.read()

        if not ret:
            break;

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(image)

    cap.release()
    cv2.destroyAllWindows()
    return frames




#얼굴 감지
# Create face detector
for video in train_fns[:]:
    print(video)
    frames = get_frames(video)
    try :
        mtcnn = MTCNN(margin=20, keep_all=False, post_process=False)  # keep_all : multiple faces in a single image,device='cuda:0'device='cuda:0'
        save_paths = [str(video) + f'_{i}.jpg' for i in range(len(frames[:241]))]
        mtcnn(frames[:241], save_path=save_paths)
    except TypeError :
        print(video + "error")
        continue