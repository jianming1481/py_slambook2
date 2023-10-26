import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from optical_flow import OpticalFlowSingleLevel

def main():
    if os.path.exists('./LK1.png'):
        image1_path = './LK1.png'
        image2_path = './LK2.png'
        # images, note they are CV_8UC1, not CV_8UC3
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    else:
        print('No file exist')

    gftt = cv2.GFTTDetector_create(maxCorners=500, qualityLevel=0.01, minDistance=20, blockSize=10, useHarrisDetector=False, k=0.1)

    keypoint1 = gftt.detect(img1, None)
    kp2_single = []
    success_single = []
    kp2_single, success_single = OpticalFlowSingleLevel(img1, img2, keypoint1)
    # # then test multi-level LK
    # kp2_multi = []
    # success_multi = []
    # t1 = time.perf_counter()
    # kp2_multi, success_multi = OpticalFlowMultiLevel(img1, img2, keypoint1, kp2_multi, success_multi, True);
    # t2 = time.perf_counter()
    # time_used = t2 - t1
    # print('optical flow by gaussian-newton: ', time_used)

if __name__ == '__main__':
    main()