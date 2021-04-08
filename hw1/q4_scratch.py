# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def imshow(*srcs):
    plt.close()
    fig, ax = plt.subplots(ncols=len(srcs), figsize=(len(srcs) * 10, 10))
    for i, src in enumerate(srcs):
        t = (src - src.min()) / (src.max() - src.min())
        if (len(srcs)) > 1:
            ax[i].imshow(t, cmap='gray')
        else:
            ax.imshow(t, cmap='gray')
    plt.show()


def apply_transform(src_points, mat):
    src_points = np.array(src_points, dtype=np.float64)
    src_points = np.hstack([src_points, np.ones(src_points.shape[0], dtype=np.float64)[:, None]])
    dst_points = np.matmul(mat, src_points.T).T
    return dst_points[:, :2] / dst_points[:, 2][:, None]


def find_homography_lsq(src_points, dst_points, mask=None, invert=True):
    if mask is None:
        mask = np.ones(src_points.shape[0], dtype=np.bool)
    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    mat = np.zeros((2 * src_points.shape[0], 9))
    for i, (pt1, pt2, val) in enumerate(zip(src_points, dst_points, mask)):
        if not val:
            continue
        if invert:
            pt2 = pt2[::-1]
        mat[2 * i] = np.array([-pt1[0], -pt1[1], -1, 0, 0, 0, pt1[0] * pt2[1], pt1[1] * pt2[1], pt2[1]])
        mat[2 * i + 1] = np.array([0, 0, 0, -pt1[0], -pt1[1], -1, pt1[0] * pt2[0], pt1[1] * pt2[0], pt2[0]])
    u, s, vh = np.linalg.svd(mat, full_matrices=True)
    return vh[-1, :].reshape((3, 3))


def find_homography_ransac(src_points, dst_points, inlier_threshold=3, max_iterations=10000, confidence=0.995):
    pass
