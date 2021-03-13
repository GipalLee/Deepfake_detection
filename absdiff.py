import pandas as pd
import numpy as np
import cv2
import glob


TRAIN_PATH = './videos/sample/'
train_fns = sorted(glob.glob(TRAIN_PATH + '*.csv'))
print('There are {} samples in the train set.'.format(len(train_fns)))

# 이미지 전체 밝기 값 평균
def calcMatrixMean(image):
    pixel_sum = 0
    size = 0
    for row in image:
        for pixel in row:
            pixel_sum += pixel
            size += 1

    return pixel_sum / size

resfile = open("abs.csv", "a")
resfile.write(
    "matrix_r,matrix_g,matrix_b,deepfake\n")
for i in range(299) :
    src1 = cv2.imread('./videos/sample/real_001.mp4_'+str(i)+'.jpg')
    src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2RGB)

    src2 = cv2.imread('./videos/sample/real_001.mp4_'+str(i+1)+'.jpg')
    src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)

    dst = cv2.absdiff(src1, src2, dst=None)

    cur = calcMatrixMean(dst)

    resfile.write(
        str(cur[0]) + ',' + str(cur[1]) + ',' + str(cur[2]) + ',' +str('0') + "\n"
    )
for i in range(299):
    src1 = cv2.imread('./videos/sample/fake_001.mp4_' + str(i) + '.jpg')
    src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2RGB)

    src2 = cv2.imread('./videos/sample/fake_001.mp4_' + str(i + 1) + '.jpg')
    src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)

    dst = cv2.absdiff(src1, src2, dst=None)

    cur = calcMatrixMean(dst)

    resfile.write(
        str(cur[0]) + ',' + str(cur[1]) + ',' + str(cur[2]) + ',' +str('1') + "\n"
    )