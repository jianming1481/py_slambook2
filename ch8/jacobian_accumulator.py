import numpy as np
import sophus as sp

class JacobianAccumulator:
    '''
    input:
        img1_: cv::Mat
        img2_: cv::Mat
        px_ref_: list[list[]]
        depth_ref_: list[list[]]
    return:
        T21_: Sophus::SE3d
    '''
    def __init__(self, img1_, img2_, px_ref_, depth_ref_):
        self.img1 = img1_
        self.img2 = img2_
        self.px_ref = px_ref_
        self.depth_ref = depth_ref_
        R_identity = sp.SO3()
        t_zero = np.zeros((3,))
        self.T21 = sp.SE3(R_identity.matrix(), t_zero)
        self.cost = 0.0
        self.projection = [[0,0] for _ in range(len(self.px_ref))]

    # Set Camera intrinsics
    def set_camera_intrinsics(self, fx, fy, cx, cy, baseline):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.base_line = baseline

    # range = slice()
    def accumulate_jacobian(self, rng):
        # Parameter
        half_patch_size = 1
        hessian = np.zeros((6, 6))
        ias = np.zeros(6)
        tmp_cost = 0.0
        cnt_good = 0
        print('accumulate_jacobian')

        for i in range(rng.start, rng.stop):
            # compute the projection in the second image
            point_ref = list(map(lambda x: self.depth_ref[i] * x, [(self.px_ref[i][0]-self.cx)/self.fx,
                                                                   (self.px_ref[i][1]-self.cy)/self.fy,
                                                                   1]))
            point_cur = self.T21 * point_ref
            if point_cur[2] < 0:        # depth invalid
                continue

            u = self.fx * point_cur[0] / point_cur[2] + self.cx
            v = self.fy * point_cur[1] / point_cur[2] + self.cy
            if u < half_patch_size or u > self.img2.shape[1] - half_patch_size or v < half_patch_size or v > self.img2.shape[0] - half_patch_size:
                continue

            self.projection[i] = [u, v]
            X = point_cur[0]
            Y = point_cur[1]
            Z = point_cur[2]
            Z2 = Z*Z
            Z_inv = 1/Z
            Z2_inv = Z_inv*Z_inv
            cnt_good += 1

            # and compute error and jacobian (11/13)

    def hessian(self):
        self.H = np.zeros((6, 6))
        return self.H

    def bias(self):
        self.b = np.zeros(6)
        return self.b

    def cost_func(self):
        return self.cost

    def projected_points(self):
        # self.projection = [[0,0],[1,1],...]
        return self.projection

    def reset(self):
        self.H = np.zeros((6, 6))
        self.b = np.zeros(6)
        self.cost = 0
