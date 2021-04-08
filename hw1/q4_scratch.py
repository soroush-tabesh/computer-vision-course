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
    src_points = np.array(src_points, dtype=np.float64)
    dst_points = np.array(dst_points, dtype=np.float64)
    if mask is None:
        mask = np.ones(src_points.shape[0], dtype=np.bool)
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


def find_homography_ransac(src_points, dst_points,
                           inlier_threshold=5, max_iterations=10000, confidence=0.995, invert=True):
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

        # mat, _ = cv.findHomography(sample_src, sample_dst)
        mat = find_homography_lsq(sample_src, sample_dst, invert=invert)
        # predicted_dst = cv.perspectiveTransform(src_points.reshape(-1, 1, 2), mat).reshape(src_points.shape)
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
