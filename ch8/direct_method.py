import cv2
import random
from jacobian_accumulator import JacobianAccumulator
import time

# Camera intrinsics
fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157
# baseline
baseline = 0.573

# DirectPoseEstimationSingleLayer return Sophus::SE3d &T21)
def DirectPoseEstimationSingleLayer(img1, img2, px_ref, depth_ref):
    iterations = 10
    cost = 0
    lastCost = 0
    t1 = time.time()
    jaco_accu = JacobianAccumulator(img1, img2, px_ref, depth_ref)

    for i in range(iterations):
        jaco_accu.reset()
        rng = slice(0, len(px_ref))
        jaco_accu.accumulate_jacobian(rng)
        H = jaco_accu.hessian()
        b = jaco_accu.bias()


def main():
    left_file = 'left.png'
    disparity_file = 'disparity.png'
    fmt_others = "./{:06d}.png"
    # file_number = 123
    # file_name = fmt_others.format(file_number)
    # print(file_name)  # 輸出 "./000123.png"
    left_img = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
    disparity_img = cv2.imread(disparity_file, cv2.IMREAD_GRAYSCALE)

    # let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    nPoint = 2000
    boarder = 20
    rows, columns = left_img.shape
    depth_ref = []
    pixels_ref = []
    for i in range(nPoint):
        x = random.randint(boarder, columns-boarder) # don't pick pixels close to boarder
        y = random.randint(boarder, rows-boarder) # don't pick pixels close to boarder
        disparity = disparity_img[y][x]
        global fx, fy, cx, cy, baseline
        depth = fx*baseline/disparity
        depth_ref.append(depth)
        pixels_ref.append([x,y])

    for i in range(1,6):
        file_number = i
        file_name = fmt_others.format(file_number)
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);



if __name__ == '__main__':
    main()