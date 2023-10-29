import numpy as np
from typing import List
import cv2

def GetPixelValue(image: np.ndarray, x: float, y: float) -> float:
    # boundary check
    yLimit, xLimit = image.shape
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > (xLimit - 1):
        x = xLimit - 2
    if y > (yLimit - 1):
        y = yLimit - 2
    # print("x: %f y: %f image[y][x]: %f" % (x, y, image[int(y)][int(x)]))
    xx = x - np.floor(x)
    yy = y - np.floor(y)
    x_a1 = np.min([xLimit - 1, int(x) + 1])
    y_a1 = np.min([yLimit - 1, int(y) + 1])
    x = int(x)
    y = int(y)
    x_a1 = int(x_a1)
    y_a1 = int(y_a1)
    # xx = int(xx)
    # yy = int(yy)
    # print('x_a1: %f y_a1: %f xx: %f yy: %f' % (x_a1, y_a1, xx, yy))
    tmp1 = (1 - xx) * (1 - yy) * image[y][x]
    tmp2 = xx * (1 - yy) * image[y][x_a1]
    tmp3 = (1 - xx) * yy * image[y_a1][x]
    tmp4 = xx * yy * image[y_a1][x_a1]
    # print('tmp1: %f tmp2: %f tmp3: %f tmp4: %f' % (tmp1,tmp2, tmp3, tmp4))
    # print('------------------------------------------------------')
    return tmp1 + tmp2 + tmp3 + tmp4

class OpticalFlowTracker:
    def __init__(self,
                 img1: np.ndarray,
                 img2: np.ndarray,
                 kp1: List[cv2.KeyPoint],
                 kp2: List[cv2.KeyPoint],
                 success: List[bool],
                 inverse: bool = True,
                 hasInit: bool = False) -> None:
        self.img1 = img1
        self.img2 = img2
        self.kp1 = kp1
        self.kp2 = kp2
        self.success = success
        self.inverse = inverse
        self.hasInit = hasInit
        print('Start OpticalFlowTracker')

    def calculateOpticalFlow(self, myRange: List[int]):
        # print('calculate optical flow')
        half_patch_size = 4
        iterations = 10
        start, end = myRange
        for i in range(start, end):
            kp = self.kp1[i]
            # if kp.pt[0] == 742 and kp.pt[1] == 340:
            #     print('stop here')
            dx = 0.0
            dy = 0.0
            if self.hasInit:
                dx = self.kp2[i].pt[0] - kp.pt[0]
                dy = self.kp2[i].pt[1] - kp.pt[1]

            cost = 0
            lastCost = 0
            succ = True  # indicate if this point succeeded
            H = np.zeros((2, 2))  # Hessian
            b = np.zeros(2)  # bias
            J = np.zeros(2)

            for myIter in range(iterations):
                if not self.inverse:
                    H = np.zeros((2, 2))  # Hessian
                    b = np.zeros(2)  # bias
                else:
                    # Only reset b
                    b = np.zeros(2)

                cost = 0

                # compute cost and Jacobian
                for x in range(-half_patch_size, half_patch_size):
                    for y in range(-half_patch_size, half_patch_size):
                        error = (GetPixelValue(self.img1, kp.pt[0] + x, kp.pt[1] + y)
                                 - GetPixelValue(self.img2, kp.pt[0] + x + dx, kp.pt[1] + y + dy))  # Jacobian
                        if not self.inverse:
                            # if kp.pt[0] == 742 and kp.pt[1] == 340:
                            #     print('changing J from ', J)
                            J = -1.0 * np.array([
                                0.5 * (GetPixelValue(self.img2, kp.pt[0] + dx + x + 1, kp.pt[1] + dy + y) - GetPixelValue(
                                    self.img2, kp.pt[0] + dx + x - 1, kp.pt[1] + dy + y)),
                                0.5 * (GetPixelValue(self.img2, kp.pt[0] + dx + x, kp.pt[1] + dy + y + 1) - GetPixelValue(
                                    self.img2, kp.pt[0] + dx + x, kp.pt[1] + dy + y - 1))
                            ])
                            # if kp.pt[0] == 742 and kp.pt[1] == 340:
                            #     print(' to ', J)
                        elif myIter == 0:
                            # in inverse mode, J keeps same for all iterations
                            # NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                            J = -1.0 * np.array([
                                0.5 * (GetPixelValue(self.img1, kp.pt[0] + x + 1, kp.pt[1] + y) - GetPixelValue(self.img1, kp.pt[0] + x - 1, kp.pt[1] + y)),
                                0.5 * (GetPixelValue(self.img1, kp.pt[0] + x, kp.pt[1] + y + 1) - GetPixelValue(self.img1, kp.pt[0] + x,kp.pt[1] + y - 1))
                            ])

                        # compute H, b and set cost;
                        b += -error * J
                        cost += error * error
                        if not self.inverse or myIter == 0:
                            # Update H
                            H += np.outer(J, J.transpose())
                # Compute update
                try:
                    # if kp.pt[0] == 742 and kp.pt[1] == 340:
                    #     print('J: ', J)
                    #     print('H: ', H)
                    #     print('b: ', b)
                    #     print('---------------------------------------')

                    try:
                        update = np.linalg.solve(H, b)
                    except:
                        H_pseudo_inv = np.linalg.pinv(H)
                        update = np.dot(H_pseudo_inv, b)

                except Exception as e:
                    print(f"An error occurred: {e}")
                    print('Iter: ', myIter)
                    print('kx: %f, ky: %f' %(kp.pt[0], kp.pt[1]))
                    print('J: ', J)
                    print('H: ', H)
                    print('b: ', b)
                    raise

                if update is None:
                    print('update is none. Fail')
                    break
                if myIter > 0 and cost > lastCost:
                    break

                # update dx, dy
                dx += update[0]
                dy += update[1]
                lastCost = cost
                succ = True
                update_norm = np.linalg.norm(update)
                if update_norm < 1e-2:
                    # converge
                    break

            self.success[i] = succ
            # set kp2
            self.kp2[i].pt[0] = kp.pt[0] + dx
            self.kp2[i].pt[1] = kp.pt[1] + dy

        return self.success, self.kp2

    def get_keyPoint2(self):
        return self.kp2

    def get_success(self):
        return self.success


def main():
    ## debug GetPixelValue function
    # import cv2
    # image1_path = './LK1.png'
    # image2_path = './LK2.png'
    # img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    # error = GetPixelValue(img1, 444.0, 388.0) - GetPixelValue(img2, 445.55377, 388.134871)
    # print(error)

    # Debug Singular Matrix
    # determind is 0 is singular matrix
    import numpy as np
    H = np.array([[0,0],[0, 13141.6376]])
    b = np.array([0, -31341.28298])
    try:
        update = np.linalg.solve(H, b)
    except:
        H_pseudo_inv = np.linalg.pinv(H)
        update = np.dot(H_pseudo_inv, b)

    print(update)

if __name__ == '__main__':
    main()