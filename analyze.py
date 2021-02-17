import json
import argparse
import glob
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

 # 히스토그램 차이
def get_image_difference(image_1, image_2):
    first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
    second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

    img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
    #img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
    #img_template_diff = 1 - img_template_probability_match

    # taking only 10% of histogram diff, since it's less accurate than template method
    #commutative_image_diff = (img_hist_diff / 10) + img_template_diff

    return img_hist_diff

# RGB 평균값 계산
def calcRGBCenter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RGB 형태로 변환
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    for i in image:
        for j in i:
            if type(j) is not int:
                print(j, type(j))
    k = 1
    clt = KMeans(n_clusters = k)
    clt.fit(image)

    return clt.cluster_centers_[0]

# HSV 평균값 계산
def calcHSVCenter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #HSV 형태로 변환
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    k = 1
    clt = KMeans(n_clusters = k)
    clt.fit(image)

    return clt.cluster_centers_[0]


# 이미지 전체 밝기 값 평균
def calcMatrixMean(image):
    pixel_sum = 0
    size = 0
    for row in image:
        for pixel in row:
            pixel_sum += pixel
            size += 1

    return pixel_sum / size


# 총엔트로피
def totalEntropy(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return skimage.measure.shannon_entropy(image)

# 총 분산
def totalVariance(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = calcMatrixMean(image)
    sum_distance = 0
    size = 0
    for row in image:
        for pixel in row:
            sum_distance += ((pixel - mean) ** 2)
            size += 1

    return sum_distance / size

# 엣지 밀집도 분석
def edgeDensityAnalysis(image):
    h, w = len(image), len(image[0])

    edge = cv2.Canny(image, 50, 130)

    total_slc = 0
    slc_include_edge = 0

    for i in range(w):
        for j in range(h):
            if edge[i][j] == 255:
                slc_include_edge += 1
            total_slc += 1

    return (slc_include_edge / total_slc) * 100


# 엣지의 불확실성

def edgeEntropy(image):
    edge = cv2.Canny(image, 50, 100)
    return skimage.measure.shannon_entropy(edge)

# dct AC계수
def dctCoefficient(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) /  1920.0
    dct = cv2.dct(image)
    image = np.uint8(dct * 1920.0)


# load the filenames for train videos
TRAIN_PATH = './videos/dfdc_train_part_00/dfdc_train_part_0/'
train_fns = sorted(glob.glob(TRAIN_PATH + '*.mp4'))
print('There are {} samples in the train set.'.format(len(train_fns)))

for i in range(0,len(train_fns)) :
    # 동영상 이름 가져오기
    videoname = train_fns[i]
    frames = []
    print(videoname)

    # 프레임 읽어서 연결
    for i in range(300):  # 가져온 동영상 얼굴 사진들로 프레임 생성
        frame = cv2.imread(videoname + '_' + str(i) + '.jpg')
        frames.append(frame)




    resfile = open(videoname + "_result.csv", "a")
    resfile.write("mse,psnr,ssim,hist_diff,r_diff,g_diff,b_diff,h_diff,s_diff,v_diff,matrix_diff_r,matrix_diff_g,matrix_diff_b, totalentropy, totalvariance, edgedensity, edgeentropy, dctcoefficient\n")

    # 프레임 간 차이 값 계산
    for i in range(len(frames) - 1):
        img1 = frames[i]  # 첫 프레임
        img2 = frames[i + 1]  # 다음 프레임

        mse = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        mse /= float(img1.shape[0] * img1.shape[1])

        psnr = cv2.PSNR(img1, img2)  # 프레임 간 PSNR값

        ssim = measure.compare_ssim(img1, img2, multichannel=True)  # SSIM 멀티채널 오픈필요

        histogram_diff = get_image_difference(img1, img2)  # 프레임간 히스토그램 차이

        rgb_diff = abs(calcRGBCenter(img2) - calcRGBCenter(img1))

        hsv_diff = abs(calcHSVCenter(img2) - calcHSVCenter(img1))

        matrix_diff = abs(calcMatrixMean(img2) - calcMatrixMean(img1))

        entropy_diff = abs(totalEntropy(img2) - totalEntropy(img1))

        variance_diff = abs(totalVariance(img2) - totalVariance(img1))

        edgedentsity_diff = abs(edgeDensityAnalysis(img2) - edgeDensityAnalysis(img1))

        edgeentropy_diff = abs(edgeEntropy(img2) - edgeEntropy(img1))

        coefficient_diff = abs(dctCoefficient(img2)[0][0] - dctCoefficient(img1)[0][0])

        resfile.write(
            str(mse) + ',' + str(psnr) + ',' + str(ssim) + ',' + str(histogram_diff) + ',' +
            str(rgb_diff[0]) + ',' + str(rgb_diff[1]) + ',' + str(rgb_diff[2]) + ',' +
            str(hsv_diff[0]) + ',' + str(hsv_diff[1]) + ',' + str(hsv_diff[2]) + ',' +
            str(matrix_diff[0]) + ',' + str(matrix_diff[1]) + ',' + str(matrix_diff[2]) + ',' +
            str(entropy_diff) + ',' + str(variance_diff) + ',' + str(edgedentsity_diff) + ',' +
            str(edgeentropy_diff) + ',' + str(coefficient_diff) + "\n"
        )
