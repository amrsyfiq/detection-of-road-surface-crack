# Test App: Detection of Road Surface Crack
# Please excuse for the brevity.  This is a test app and not for Production.

import cv2
# import datetime
import numpy as np
import skimage.io as io
# from skimage import data_dir
from matplotlib import pyplot as plt

# You must add the picture with the main.py at the same path
str = 'pic/*.jpg'
coll = io.ImageCollection(str)
# number = [0]*len(coll)

for i in range(0, len(coll)):
    r1 = cv2.pyrDown(coll[i])
    r2 = cv2.pyrDown(r1)
    r3 = cv2.pyrDown(r2)

    GrayImage = cv2.cvtColor(r3, cv2.COLOR_BGR2GRAY)
    dst = cv2.GaussianBlur(GrayImage, (5, 5), 0, 0)
    img = dst
    cv2.imshow("Input-set", img)
    cv2.waitKey(0)
    ret, thresh5 = cv2.threshold(img, 145, 255, cv2.THRESH_TOZERO_INV)
    k = np.ones((4, 4), np.uint8)
    erosion = cv2.morphologyEx(thresh5, cv2.MORPH_OPEN, k)
    k_close = np.ones((4, 4), np.uint8)
    Erosion = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, k_close)
    cv2.imshow("Output-Set", Erosion)
    cv2.waitKey(0)
    # #----------------------------------------------------------------------------#
    # brisk = cv2.BRISK_create()
    # keypoints = brisk.detect(Erosion, None)
    # print("Start of Value")
    # # ------------------------------------------------------#
    # print("Start")
    # img2 = Erosion
    # img2 = cv2.drawKeypoints(Erosion, keypoints, img2, color=(0, 255, 0))
    # cv2.imshow('Detected BRISK keypoints', img2)
    # cv2.waitKey(0)
    # #------------------------------------------------------------------------#
    # starttime = datetime.datetime.now()
    # fast = cv2.FastFeatureDetector_create(50)
    # kp = fast.detect(Erosion, None)
    # img2 = cv2.drawKeypoints(Erosion, kp, (0, 255, 0))
    # endtime = datetime.datetime.now()
    # a = endtime - starttime
    # print('number=', len(kp))
    # number[i] = len(kp)
    # ava = np.sum(dst) / np.size(dst)
    # print('average=', ava)
    # cv2.namedWindow('fast', cv2.WINDOW_NORMAL)

    # cv2.imshow('fast', img2)
    # cv2.waitKey(0)
    # if len(kp) >= 45 and len(kp) <= 145:
    #     print('have')
    # else:
    #     print('No')
    # print('#----------------------------------------------#')

    # print all_number/len(coll)
    fast = cv2.FastFeatureDetector_create(
        threshold=70, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    keypoints = fast.detect(Erosion, None)
    print("nonmaxSuppression: ", fast.getNonmaxSuppression())
    print('FastFeatureDetector_create')
    print("NumberOfKeypoint: ", len(keypoints))
    if len(keypoints) >= 30:
        print('Crack surface detected')
    else:
        print('Crack surface not detected')
    print('#----------------------------------------------#')

    img2 = Erosion
    img2 = cv2.drawKeypoints(img2, keypoints, img2, color=(0, 255, 0))
    cv2.imshow('Detected FAST keypoints', img2)
    cv2.waitKey(0)

    # Use plot to show input and output image
    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img2)
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    plt.show()
