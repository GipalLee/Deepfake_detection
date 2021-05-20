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

    rgb_data = []
    rgb_avg = []
    rgb_max = []

    Red = []
    Green = []
    Blue = []

    for x in image:
        for y in x:
            Red.append(y[0])
            Green.append(y[1])
            Blue.append(y[2])

    R_avg = sum(Red) / len(Red)
    G_avg = sum(Green) / len(Green)
    B_avg = sum(Blue) / len(Blue)

    R_max = max(Red)
    G_max = max(Green)
    B_max = max(Blue)

    rgb_avg.append(R_avg)
    rgb_avg.append(G_avg)
    rgb_avg.append(B_avg)

    rgb_max.append(R_max)
    rgb_max.append(G_max)
    rgb_max.append(B_max)

    rgb_data.append(rgb_avg)
    rgb_data.append(rgb_max)

    rgb_data= np.array(rgb_data)
    return rgb_data

# HSV 평균값 계산
def calcHSVCenter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #HSV 형태로 변환

    hsv_data = []
    hsv_avg = []
    hsv_max = []

    h = []
    s = []
    v = []

    for x in image:
        for y in x:
            h.append(y[0])
            s.append(y[1])
            v.append(y[2])

    h_avg = sum(h) / len(h)
    s_avg = sum(s) / len(s)
    v_avg = sum(v) / len(v)

    h_max = max(h)
    s_max = max(s)
    v_max = max(v)

    hsv_avg.append(h_avg)
    hsv_avg.append(s_avg)
    hsv_avg.append(v_avg)

    hsv_max.append(h_max)
    hsv_max.append(s_max)
    hsv_max.append(v_max)

    hsv_data.append(hsv_avg)
    hsv_data.append(hsv_max)

    hsv_data = np.array(hsv_data)
    return hsv_data

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


# 엣지 노이즈 분석

def edgeNoiseAnalysis(image):
    h, w = len(image), len(image[0])

    edge = cv2.Canny(image, 50, 130)

    base = cv2.getGaussianKernel(5, 5)
    kernel = np.outer(base, base.transpose())

    arr = cv2.filter2D(edge, -1, kernel)

    edgecount = 0
    arrcount = 0

    for i in range(w):
        for j in range(h):
            if arr[i][j] < 55:
                arr[i][j] = 0
            else:
                arr[i][j] = 255
                arrcount += 1

            if edge[i][j] == 255:
                edgecount += 1

    return (arrcount - edgecount) / edgecount * 100


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

    return image

TRAIN_PATH = './dataset/faceswap/'
train_fns = sorted(glob.glob(TRAIN_PATH + '*.mp4'))
print('There are {} samples in the train set.'.format(len(train_fns)))

# video data extraction
for i in range(len(train_fns)):

    # 동영상 이름 가져오기
    videoname = train_fns[i]
    frames = []
    print(videoname)

    # 프레임 읽어서 연결
    for i in range(240):  # 가져온 동영상 얼굴 사진들로 프레임 생성
        frame = cv2.imread(videoname + '_' + str(i) + '.jpg')
        frames.append(frame)

    resfile = open(videoname + "_result.csv", "a")
    resfile.write(
        "mse,psnr,ssim,hist_diff,r_avg_diff,g_avg_diff,b_avg_diff,r_max_diff,g_max_diff,b_max_diff,h_avg_diff,s_avg_diff,v_avg_diff,h_max_diff,s_max_diff,v_max_diff,matrix_diff_r,matrix_diff_g,matrix_diff_b, totalentropy, totalvariance, edgedensity, edgeentropy, dctcoefficient\n")

    for i in range(len(frames) - 1):
        print(i)
        img1 = frames[i]  # 첫 프레임
        img2 = frames[i + 1]  # 다음 프레임

        mse = np.sum((img1.astype("float") - img2.astype("float")) ** 2)  # MSE
        mse /= float(img1.shape[0] * img1.shape[1])

        psnr = cv2.PSNR(img1, img2)  # 프레임 간 PSNR값

        ssim = measure.compare_ssim(img1, img2, multichannel=True)  # SSIM 멀티채널 오픈필요

        histogram_diff = get_image_difference(img1, img2)  # 프레임간 히스토그램 차이

        rgb_avg_diff = abs(calcRGBCenter(img2)[0] - calcRGBCenter(img1)[0])

        rgb_max_diff = abs(calcRGBCenter(img2)[1] - calcRGBCenter(img1)[1])

        hsv_avg_diff = abs(calcHSVCenter(img2)[0] - calcHSVCenter(img1)[0])

        hsv_max_diff = abs(calcHSVCenter(img2)[1] - calcHSVCenter(img1)[1])


        matrix_diff = abs(calcMatrixMean(img2) - calcMatrixMean(img1))

        entropy_diff = abs(totalEntropy(img2) - totalEntropy(img1))

        variance_diff = abs(totalVariance(img2) - totalVariance(img1))

        edgedentsity_diff = abs(edgeDensityAnalysis(img2) - edgeDensityAnalysis(img1))

        edgeentropy_diff = abs(edgeEntropy(img2) - edgeEntropy(img1))

        coefficient_diff = abs(float(dctCoefficient(img2)[0][0]) - float(dctCoefficient(img1)[0][0]))

        resfile.write(
            str(mse) + ',' + str(psnr) + ',' + str(ssim) + ',' + str(histogram_diff) + ',' +
            str(rgb_avg_diff[0]) + ',' + str(rgb_avg_diff[1]) + ',' + str(rgb_avg_diff[2]) + ',' +
            str(rgb_max_diff[0]) + ',' + str(rgb_max_diff[1]) + ',' + str(rgb_max_diff[2]) + ',' +
            str(hsv_avg_diff[0]) + ',' + str(hsv_avg_diff[1]) + ',' + str(hsv_avg_diff[2]) + ',' +
            str(hsv_max_diff[0]) + ',' + str(hsv_max_diff[1]) + ',' + str(hsv_max_diff[2]) + ',' +
            str(matrix_diff[0]) + ',' + str(matrix_diff[1]) + ',' + str(matrix_diff[2]) + ',' +
            str(entropy_diff) + ',' + str(variance_diff) + ',' +
            str(edgedentsity_diff) + ',' + str(edgeentropy_diff) + ',' + str(coefficient_diff) + '\n'
        )

