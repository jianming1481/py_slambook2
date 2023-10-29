import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from optical_flow import OpticalFlowSingleLevel
from optical_flow import OpticalFlowMultiLevel
import time

def main():
    if os.path.exists('./LK1.png'):
        image1_path = './LK1.png'
        image2_path = './LK2.png'
        # images, note they are CV_8UC1, not CV_8UC3
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    else:
        print('No file exist')

    gftt = cv2.GFTTDetector_create(maxCorners=500, qualityLevel=0.01, minDistance=20)
    # gftt = cv2.GFTTDetector_create(maxCorners=500, qualityLevel=0.01, minDistance=20, blockSize=10, useHarrisDetector=False, k=0.1)

    keypoint1 = gftt.detect(img1, None)

    '''
    # # 繪製角點
    # output_image = cv2.drawKeypoints(img1, keypoint1, None, color=(0, 255, 0))
    #
    # # 顯示圖像
    # cv2.imshow('Detected Keypoints', output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    '''

    # Single level LK in the validation picture
    keypoint2Dummy = [] # Dummy Keypoint2 (Keypoint2 won't be used in Single Level LK)
    successDummy = []
    t1 = time.perf_counter()
    kp2_single, success_single = OpticalFlowSingleLevel(img1, img2, keypoint1, keypoint2Dummy, successDummy)
    t2 = time.perf_counter()
    time_used = t2 - t1
    print('optical flow Single Layer: ', time_used)

    # 將 img2 轉換為 BGR 彩色圖像
    img2_single = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # 繪製關鍵點和連接線
    for i in range(len(kp2_single)):
        if success_single[i]:
            # 畫出圓形來表示關鍵點
            cv2.circle(img2_single, (int(kp2_single[i].pt[0]), int(kp2_single[i].pt[1])), 2, (0, 250, 0), 2)

            # 畫出從 kp1 到 kp2_single 的線
            cv2.line(img2_single,
                     (int(keypoint1[i].pt[0]), int(keypoint1[i].pt[1])),
                     (int(kp2_single[i].pt[0]), int(kp2_single[i].pt[1])),
                     (0, 250, 0))


    # then test multi-level LK
    t1 = time.perf_counter()
    kp2_multi, success_multi = OpticalFlowMultiLevel(img1, img2, keypoint1, True)
    t2 = time.perf_counter()
    time_used = t2 - t1
    print('optical flow by gaussian-newton: ', time_used)


    # 將 img2 轉換為 BGR 彩色圖像
    img2_multi = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # 繪製關鍵點和連接線
    for i in range(len(kp2_multi)):
        if success_multi[i]:
            # 畫出圓形來表示關鍵點
            cv2.circle(img2_multi, (int(kp2_multi[i].pt[0]), int(kp2_multi[i].pt[1])), 2, (0, 250, 0), 2)

            # 畫出從 kp1 到 kp2_multi 的線
            cv2.line(img2_multi,
                     (int(keypoint1[i].pt[0]), int(keypoint1[i].pt[1])),
                     (int(kp2_multi[i].pt[0]), int(kp2_multi[i].pt[1])),
                     (0, 250, 0))

    # OpenCV Version Optical Flow
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = np.array([kp.pt for kp in keypoint1], dtype=np.float32).reshape(-1, 1, 2)
    t1 = time.perf_counter()
    p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, ** lk_params)
    t2 = time.perf_counter()
    time_used = t2 - t1
    print('optical flow by opencv: ', time_used)
    # 選擇好的點
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 繪圖
    img2_opencv = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = map(int, new.ravel())  # 確保座標是整數
        c, d = map(int, old.ravel())  # 確保座標是整數
        cv2.line(img2_opencv, (a, b), (c, d), (0, 255, 0))
        cv2.circle(img2_opencv, (a, b), 2, (0, 255, 0), 2)

    # 顯示結果圖像（可選）
    cv2.imshow('tracked single level (Python)', img2_single)
    cv2.imshow('tracked multi level  (Python)', img2_multi)
    cv2.imshow('tracked by opencv  (Python)', img2_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()