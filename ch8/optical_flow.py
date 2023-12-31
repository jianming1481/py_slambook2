import numpy as np
import cv2
from typing import List
from typing import Tuple
from optical_flow_tracker import OpticalFlowTracker
import time

'''
 single level optical flow
 return keypoint2: List[cv2.KeyPoint], success: List[bool]
'''
class MyPoint:
    def __init__(self, x, y, size = 0):
        self.pt = np.zeros(2)
        self.pt[0] = x
        self.pt[1] = y
        self.size = size

    def __setitem__(self, index, value):
        if index == 0 or index == 1:
            self.pt[index] = value
        else:
            raise IndexError("Invalid index")


def OpticalFlowSingleLevel(
        img1: np.ndarray,
        img2: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[MyPoint],
        success: List[bool],
        inverse: bool = False,
        has_initial_guess: bool = False) -> Tuple[List[cv2.KeyPoint], List[bool]]:

    if not has_initial_guess or len(kp2) == 0:
        kp2 = []
        for i in range(len(kp1)):
            kp = MyPoint(0,0)
            kp2.append(kp)

    if len(success) == 0:
        success = []
        for i in range(len(kp1)):
            success.append(False)

    tracker = OpticalFlowTracker(img1, img2, kp1, kp2, success, inverse, has_initial_guess)

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #   # executor.map(tracker.calculateOpticalFlow, range(len(kp1))
    #   futures = [executor.submit(tracker.calculate_optical_flow, i) for i in range(len(kp1))]
    #   results = [future.result() for future in concurrent.futures.as_completed(futures)]
    myRange = [0, len(kp1)]
    success, kp2 = tracker.calculateOpticalFlow(myRange)
    return kp2, success

'''
 multi level optical flow, scale of pyramid is set to 2 by default
 the image pyramid will be create inside the function
 return kp1, kp2, success
'''
def OpticalFlowMultiLevel(
        img1: np.ndarray,
        img2: np.ndarray,
        kp1: List[cv2.KeyPoint],
        inverse: bool = False, ) -> Tuple[List[cv2.KeyPoint], List[bool]]:
    print('OpticalFlowMultiLevel')
    # parameters
    pyramids = 4
    pyramid_scale = 0.5
    scales = [1.0, 0.5, 0.25, 0.125]
    # create pyramids
    t1 = time.perf_counter()
    pyr1 = []
    pyr2 = []
    for i in range(pyramids):
        if i == 0:
            pyr1.append(img1)
            pyr2.append(img2)
        else:
            new_width = int(pyr1[i - 1].shape[1] * pyramid_scale)
            new_height = int(pyr1[i - 1].shape[0] * pyramid_scale)
            img1_pyr = cv2.resize(pyr1[i - 1], (new_width, new_height))
            new_width = int(pyr2[i - 1].shape[1] * pyramid_scale)
            new_height = int(pyr2[i - 1].shape[0] * pyramid_scale)
            img2_pyr = cv2.resize(pyr2[i - 1], (new_width, new_height))
            # cv2.imshow('image1 pyramid', img1_pyr)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imshow('image2 pyramid', img2_pyr)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            pyr1.append(img1_pyr)
            pyr2.append(img2_pyr)
    t2 = time.perf_counter()
    time_used = t2 - t1
    print('build pyramid time: ', time_used)
    # coarse-to-fine LK tracking in pyramids
    kp1_pyr = []
    kp2_pyr = []
    success = []
    for kp in kp1:
        kp_top = cv2.KeyPoint(kp.pt[0] * scales[pyramids - 1], kp.pt[1] * scales[pyramids - 1], kp.size)
        kp1_pyr.append(kp_top)
        kp2_top = MyPoint(kp.pt[0] * scales[pyramids - 1],kp.pt[1] * scales[pyramids - 1], kp.size)
        kp2_pyr.append(kp2_top)
        success.append(False)

    for level in range(pyramids - 1, -1, -1):
        t1 = time.perf_counter()
        kp2_pyr, success = OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, True)
        t2 = time.perf_counter()
        time_used = t2 - t1
        print('track pyr %d cost time %f' % (level, time_used))
        if level > 0:
            for i in range(len(kp1_pyr)):
                kp_x = kp1_pyr[i].pt[0] / pyramid_scale
                kp_y = kp1_pyr[i].pt[1] / pyramid_scale
                kp_size = kp1_pyr[i].size
                new_kp = cv2.KeyPoint(kp_x, kp_y, kp_size)
                kp1_pyr[i] = new_kp

            for i in range(len(kp2_pyr)):
                kp_x = kp2_pyr[i].pt[0] / pyramid_scale
                kp_y = kp2_pyr[i].pt[1] / pyramid_scale
                kp_size = kp2_pyr[i].size
                new_kp = MyPoint(kp_x, kp_y, kp_size)
                kp2_pyr[i] = new_kp

    kp2 = []
    for kp in kp2_pyr:
        kp2.append(kp)

    return kp2, success