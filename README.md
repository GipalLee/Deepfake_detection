# Deepfake detection

![version](https://img.shields.io/badge/version-1.0rc1-blue) ![version](https://img.shields.io/badge/status-rc-blue) ![version](https://img.shields.io/badge/build-passing-green) ![version](https://img.shields.io/badge/license-MIT-blue)

## Motivation

Deepfakes can be exploited for political abuse, pornography, and fake information.
AEGuard is an adversarial sample detection model based on edge noise features, which commonly appears in adversarial samples.

## Tech/framework used

* Tensorflow
* OpenCV

## Requirements

* Python 3
  * IPython 7.18.1 or greater
  * Tensorflow 2.3.0 or greater
  * Keras 2.4.3 or greater
  * Scikit-image 0.17.2 or greater
  * Scikit-learn 0.23.2 or greater
  * Numpy 1.18.5 or greater
  * Pandas 1.1.3 or greater
* OpenCV 4

## Proposed System
![image](https://user-images.githubusercontent.com/55551567/118912037-25f9e600-b962-11eb-8498-be8c79b87422.png)

The method is divided into a preprocessing process and a classification process. The pre-processing process extracts a face image from a frame image, extracts computer vision features, and then extracts the difference between frames. The classification process de-tects a deepfake through DNN by obtaining a variance of a certain number of frames from the data that has passed through the previous process.

## Features
![image](https://user-images.githubusercontent.com/55551567/118912273-7f621500-b962-11eb-889b-ffda140ba2d4.png)

The extracted features are shown in Table 2. mse(mean squared error) measures the simi-larity of an image using the difference in intensity of pixels between two images. psnr(peak signal-to-noise ratio) evaluates loss information for image quality. psnr focuses on numerical differences rather than human visual differences. Since psnr is calculated using mse, when mse is 0, psnr is also set to 0. ssim(structural similarity index measure) evaluates the temporal difference felt by humans in terms of luminance, contrast, and structural aspects. rgb(red, green, blue) and hsv(hue, saturation, value) represents the color space of an image. Histogram represents the distribution of hues in the image. Luminance represents the average of the total brightness of the image. Variance represents the vari-ance of image brightness values. edge_density is the ratio of the edge component of all pixels. dct(discrete cosine transform) refers to the sharpness of an image. Since the deep-fake production method synthesizes the target image for each frame, it may give unnatu-ral changes to various computer vision features. In addition, when making a deepfake, the target image is taken with limited resolution, and the size is changed as transformation matrices to fit the source image, so the sharpness is often inferior. Also, distortion and blur occur. Selected features are greatly influenced the deepfake creation process.

## Frames showing a significant rate of change
![image](https://user-images.githubusercontent.com/55551567/118912104-3f029700-b962-11eb-9d6b-e67eb245bd4b.png)




## Performance
![image](https://user-images.githubusercontent.com/55551567/118912468-cf40dc00-b962-11eb-83cd-363f6c198609.png)



