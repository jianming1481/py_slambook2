import cv2
import random

# Camera intrinsics
fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157
# baseline
baseline = 0.573

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



if __name__ == '__main__':
    main()