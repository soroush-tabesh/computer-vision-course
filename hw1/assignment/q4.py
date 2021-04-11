# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math


def apply_transform(src_points, mat):
    src_points = np.array(src_points, dtype=np.float64)
    src_points = np.hstack([src_points, np.ones(src_points.shape[0], dtype=np.float64)[:, None]])
    dst_points = np.matmul(mat, src_points.T).T
    return dst_points[:, :2] / (dst_points[:, 2][:, None] + 0.000001)


def normalizer_2d(points):
    points = np.array(points, dtype=np.float64)
    mean = np.average(points, axis=0)
    std = np.std(points, axis=0)
    std = np.maximum(std, np.ones_like(std) / 100)
    mat = np.array([[1 / std[0] * math.sqrt(2), 0, -mean[0] / std[0] * math.sqrt(2)],
                    [0, 1 / std[1] * math.sqrt(2), -mean[1] / std[1] * math.sqrt(2)],
                    [0, 0, 1]])
    return apply_transform(points, mat), mat


def find_homography_lsq(src_points, dst_points, mask=None, invert=True):
    src_points = np.array(src_points, dtype=np.float64)
    dst_points = np.array(dst_points, dtype=np.float64)
    if mask is None:
        mask = np.ones(src_points.shape[0], dtype=np.bool)
    src_points, t1 = normalizer_2d(src_points)
    dst_points, t2 = normalizer_2d(dst_points)
    mat = np.zeros((2 * src_points.shape[0], 9))
    for i, (pt1, pt2, val) in enumerate(zip(src_points, dst_points, mask)):
        if not val:
            continue
        if invert:
            pt2 = pt2[::-1]
        mat[2 * i] = np.array([-pt1[0], -pt1[1], -1, 0, 0, 0, pt1[0] * pt2[1], pt1[1] * pt2[1], pt2[1]])
        mat[2 * i + 1] = np.array([0, 0, 0, -pt1[0], -pt1[1], -1, pt1[0] * pt2[0], pt1[1] * pt2[0], pt2[0]])
    u, s, vh = np.linalg.svd(mat, full_matrices=True)
    h = vh[-1, :].reshape((3, 3))
    h = np.matmul(h, t1)
    h = np.matmul(np.linalg.inv(t2), h)
    return h


def find_homography_ransac(src_points, dst_points,
                           inlier_threshold=5, max_iterations=10000, confidence=0.8, invert=True):
    src_points = np.array(src_points, dtype=np.float64)
    dst_points = np.array(dst_points, dtype=np.float64)

    best_sample_size = 0
    best_sample_inliers = None

    for _it in range(max_iterations):
        if best_sample_size / src_points.shape[0] > confidence:
            break

        sample_idx = np.random.randint(0, src_points.shape[0], 4)
        sample_src = src_points[tuple(sample_idx), :]
        sample_dst = dst_points[tuple(sample_idx), :]

        mat = find_homography_lsq(sample_src, sample_dst, invert=invert)
        predicted_dst = apply_transform(src_points, mat)
        err_vec = np.linalg.norm(predicted_dst - dst_points, axis=1)

        inliers = err_vec < inlier_threshold
        size = np.sum(inliers)
        if size > best_sample_size:
            best_sample_size = size
            best_sample_inliers = inliers
    src_final_points = np.array([p for i, p in enumerate(src_points) if best_sample_inliers[i]], dtype=np.float64)
    dst_final_points = np.array([p for i, p in enumerate(dst_points) if best_sample_inliers[i]], dtype=np.float64)
    return find_homography_lsq(src_final_points, dst_final_points), best_sample_inliers


# %%
img1 = plt.imread('./data/hw1/im03.jpg')
img2 = plt.imread('./data/hw1/im04.jpg')

# %%
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# %%
match_ratio_threshold = 0.7
matches = cv.BFMatcher().knnMatch(des1, des2, k=2)
matches = [m1 for m1, m2 in matches if m1.distance < match_ratio_threshold * m2.distance]

# %%
img1_final_points = np.float64([kp1[m.queryIdx].pt for m in matches])
img2_final_points = np.float64([kp2[m.trainIdx].pt for m in matches])

M, mask = find_homography_ransac(img2_final_points, img1_final_points, max_iterations=5000, inlier_threshold=5)
img2_warped = cv.warpPerspective(img2, M, img1.shape[:2][::-1])

plt.imsave('./out/res20.jpg', img2_warped)
