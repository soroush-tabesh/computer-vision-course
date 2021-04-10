# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import util

img1 = util.img_as_float64(plt.imread('./data/hw1/im01.jpg'))
img2 = util.img_as_float64(plt.imread('./data/hw1/im02.jpg'))


def normalized(src):
    return (src - src.min()) / (src.max() - src.min())


# %% grad

def get_derivatives(src, k_size=3, mag=False):
    d_x = cv.Sobel(src, cv.CV_64F, 1, 0, ksize=k_size, borderType=cv.BORDER_REFLECT101)
    d_y = cv.Sobel(src, cv.CV_64F, 0, 1, ksize=k_size, borderType=cv.BORDER_REFLECT101)
    if mag:
        return np.sqrt(d_x ** 2 + d_y ** 2)
    else:
        return d_x, d_y


img1_d_mag = get_derivatives(img1, mag=True)
img2_d_mag = get_derivatives(img2, mag=True)

plt.imsave('./out/res01_grad.jpg', normalized(img1_d_mag))
plt.imsave('./out/res02_grad.jpg', normalized(img2_d_mag))


# %% harris score

def get_structure_tensor(src, sigma):
    d_x, d_y = get_derivatives(src)
    d2_x = d_x ** 2
    d2_y = d_y ** 2
    d_xy = d_x * d_y

    k_size = (int(4 * sigma + 1), int(4 * sigma + 1))
    s2_x = cv.GaussianBlur(d2_x, k_size, sigma)
    s2_y = cv.GaussianBlur(d2_y, k_size, sigma)
    s_xy = cv.GaussianBlur(d_xy, k_size, sigma)

    return s2_x, s_xy, s2_y


def calc_harris_score(src, sigma, k):
    s2_x, s_xy, s2_y = get_structure_tensor(src, sigma)

    det = s2_x * s2_y - s_xy ** 2
    tr = s2_x + s2_y

    return det - k * tr ** 2


sigma = 5
k = 0.05
img1_harris_score = calc_harris_score(img1, sigma, k)
img2_harris_score = calc_harris_score(img2, sigma, k)

plt.imsave('./out/res03_score.jpg', normalized(img1_harris_score))
plt.imsave('./out/res04_score.jpg', normalized(img2_harris_score))

# %% thresh

thresh = 0.01
img1_harris_score_thresh = np.where(img1_harris_score > thresh, img1_harris_score, 0)
img2_harris_score_thresh = np.where(img2_harris_score > thresh, img2_harris_score, 0)

plt.imsave('./out/res05_thresh.jpg', normalized(img1_harris_score_thresh))
plt.imsave('./out/res06_thresh.jpg', normalized(img2_harris_score_thresh))


# %% harris

def non_max_suppression(src, min_dist=50, thresh=0.0001):
    src = src.copy()
    coords = []
    while src.max() > thresh:
        coord = np.argmax(src)
        coord = np.unravel_index(coord, src.shape)
        coords.append(coord)
        src[max(0, coord[0] - min_dist):min(src.shape[0], coord[0] + min_dist),
        max(0, coord[1] - min_dist):min(src.shape[1], coord[1] + min_dist)] = 0

    return np.array(coords)[..., :2]


def draw_marker(src, points, marker_size=20, thickness=3):
    src = src.copy()
    for pt in points:
        cv.drawMarker(src, tuple(pt[::-1]), color=(0, 1, 0), markerType=0, markerSize=marker_size, thickness=thickness)
    return src


min_dist = 50
img1_harris_points = non_max_suppression(img1_harris_score_thresh, min_dist, thresh)
img2_harris_points = non_max_suppression(img2_harris_score_thresh, min_dist, thresh)

img1_harris = draw_marker(img1, img1_harris_points)
img2_harris = draw_marker(img2, img2_harris_points)

plt.imsave('./out/res07_harris.jpg', img1_harris)
plt.imsave('./out/res08_harris.jpg', img2_harris)


# %%

def get_neighborhood(src, pt, dist):
    return src[max(0, pt[0] - dist):min(src.shape[0], pt[0] + dist),
           max(0, pt[1] - dist):min(src.shape[1], pt[1] + dist)]


def get_feature_vector(src, pt):
    return get_neighborhood(src, pt, 30).ravel()


def get_features_distance(fv1, fv2):
    return cv.compareHist(fv1.astype(np.float32), fv2.astype(np.float32), cv.HISTCMP_CHISQR_ALT)


img1_feature_vectors = [get_feature_vector(img1, pt) for pt in img1_harris_points]
img2_feature_vectors = [get_feature_vector(img2, pt) for pt in img2_harris_points]

dist = np.array([[get_features_distance(feature_vector1, feature_vector2)
                  for feature_vector2 in img2_feature_vectors]
                 for feature_vector1 in img1_feature_vectors])

match_threshold = 0.8

match_for_img1 = -np.ones(dist.shape[0], dtype=int)
match_for_img2 = -np.ones(dist.shape[1], dtype=int)

for i in range(dist.shape[0]):
    srt = np.argsort(dist[i, :])
    p1, p2 = srt[0], srt[1]
    d1, d2 = dist[i, [p1, p2]]
    if d1 / d2 < match_threshold:
        match_for_img1[i] = p1

for i in range(dist.shape[1]):
    srt = np.argsort(dist[:, i])
    p1, p2 = srt[0], srt[1]
    d1, d2 = dist[[p1, p2], i]
    if d1 / d2 < match_threshold:
        match_for_img2[i] = p1

img1_corres = img1.copy()
img2_corres = img2.copy()

for i, m in enumerate(match_for_img1):
    if m != -1 and match_for_img2[m] == i:
        cv.drawMarker(img1_corres, tuple(img1_harris_points[i][::-1]),
                      color=(1, 0, 0), markerType=0, markerSize=20, thickness=3)
        cv.drawMarker(img2_corres, tuple(img2_harris_points[m][::-1]),
                      color=(1, 0, 0), markerType=0, markerSize=20, thickness=3)

plt.imsave('./out/res09_corres.jpg', img1_corres)
plt.imsave('./out/res10_corres.jpg', img2_corres)

# %%
img_side_by_side = np.concatenate([img1_corres, img2_corres], axis=1)
for i, m in enumerate(match_for_img1):
    if m != -1 and match_for_img2[m] == i:
        cv.line(img_side_by_side, tuple(img1_harris_points[i][::-1]),
                tuple(img2_harris_points[m][::-1] + np.array([img1.shape[1], 0]))
                , color=(0, 1, 0), thickness=1)

plt.imsave('./out/res11.jpg', img_side_by_side)
