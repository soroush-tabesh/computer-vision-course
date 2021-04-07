# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import util, filters, color, feature, morphology


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


img1 = util.img_as_float64(plt.imread('./data/hw1/im01.jpg'))
img2 = util.img_as_float64(plt.imread('./data/hw1/im02.jpg'))

imshow(img1, img2)

img1_gray = color.rgb2gray(img1)
img2_gray = color.rgb2gray(img2)


# %%

def sk_harris(src):
    src = src.copy()
    src_gray = color.rgb2gray(src)
    harris = feature.corner_harris(src_gray, sigma=5)
    coords = feature.corner_peaks(harris, min_distance=50, threshold_rel=0.02)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(src, cmap='gray')
    ax.plot(coords[:, 1], coords[:, 0], color='red', marker='+',
            linestyle='None', markersize=8)
    plt.show()


sk_harris(img1)
sk_harris(img2)

# %%
tt = feature.corner_harris(img1_gray, sigma=5)
imshow(tt)
plt.hist(np.ravel(tt), log='y')
plt.axvline(filters.threshold_otsu(tt), color='r')
plt.show()


# %%

def get_derivatives(src, k_size=3):
    d_x = cv.Sobel(src, cv.CV_64F, 1, 0, ksize=k_size, borderType=cv.BORDER_REFLECT101)
    d_y = cv.Sobel(src, cv.CV_64F, 0, 1, ksize=k_size, borderType=cv.BORDER_REFLECT101)
    return d_x, d_y


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


def calc_harris(src, sigma, k):
    s2_x, s_xy, s2_y = get_structure_tensor(src, sigma)

    det = s2_x * s2_y - s_xy ** 2
    tr = s2_x + s2_y

    return det - k * tr ** 2


tt = calc_harris(img1_gray, sigma=5, k=0.05)
imshow(tt, np.where(tt > 0.02, tt, 0))
plt.hist(np.ravel(tt), log='y')
plt.axvline(filters.threshold_otsu(tt), color='r')
plt.show()

# %%
d_x, d_y = get_derivatives(img1)
ff = np.sqrt(d_x ** 2 + d_y ** 2)
ff /= ff.max()
plt.imsave('chert.jpg', ff)


# %%

def non_max_suppression(src, min_dist=50, thresh=0.0001):
    src = src.copy()
    coords = []
    while src.max() > thresh:
        coord = np.argmax(src)
        coord = np.unravel_index(coord, src.shape)
        coords.append(coord)
        src[max(0, coord[0] - min_dist):min(src.shape[0], coord[0] + min_dist),
        max(0, coord[1] - min_dist):min(src.shape[1], coord[1] + min_dist)] = 0

    return np.array(coords)


def my_harris(src, sigma=5.0, k=0.05, min_dist=50, thresh=0.01, show=False):
    harris = calc_harris(color.rgb2gray(src), sigma, k)
    coords = non_max_suppression(harris, min_dist=min_dist, thresh=thresh)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(src, cmap='gray')
        plt.plot(coords[:, 1], coords[:, 0], color='r', marker='+',
                 linestyle='None', markersize=8)
        plt.show()
    return coords


my_harris(img1, thresh=0.01, show=True)
my_harris(img2, thresh=0.01, show=True)


# %%
def get_neighborhood(src, pt, dist):
    return src[max(0, pt[0] - dist):min(src.shape[0], pt[0] + dist),
           max(0, pt[1] - dist):min(src.shape[1], pt[1] + dist)]


def get_feature_vector(src, pt):
    return get_neighborhood(src, pt, 30).ravel()

    # lbp = feature.local_binary_pattern(color.rgb2gray(get_neighborhood(src, pt, 40)), 16, 2, method='uniform')
    # hs, cn = np.unique(lbp.ravel(), return_counts=True)
    # res = np.zeros(30)
    # res[np.asarray(hs, dtype=int)] = cn
    # res /= res.sum()
    # return res

    # img = get_neighborhood(src, pt, 30)
    # return np.concatenate([np.histogram(img[:, :, i].ravel(), 50, (0.0, 1.0), density=True)[0] for i in range(3)])


def get_features_distance(fv1, fv2):
    return cv.compareHist(fv1.astype(np.float32), fv2.astype(np.float32), cv.HISTCMP_CHISQR_ALT)


fpts1 = my_harris(img1, 5, 0.05, 30, 0.01, False)
fvs1 = [get_feature_vector(img1, fpt) for fpt in fpts1]

fpts2 = my_harris(img2, 5, 0.05, 30, 0.01, False)
fvs2 = [get_feature_vector(img2, fpt) for fpt in fpts2]

dist = np.zeros((len(fvs1), len(fvs2)))
for i, fv1 in enumerate(fvs1):
    for j, fv2 in enumerate(fvs2):
        dist[i, j] = get_features_distance(fv1, fv2)

thresh_mtc = 0.8

mtc1 = -np.ones(dist.shape[0], dtype=int)
mtc2 = -np.ones(dist.shape[1], dtype=int)

for i in range(dist.shape[0]):
    srt = np.argsort(dist[i, :])
    p1, p2 = srt[0], srt[1]
    d1, d2 = dist[i, [p1, p2]]
    if d1 / d2 < thresh_mtc:
        mtc1[i] = p1

for i in range(dist.shape[1]):
    srt = np.argsort(dist[:, i])
    p1, p2 = srt[0], srt[1]
    d1, d2 = dist[[p1, p2], i]
    if d1 / d2 < thresh_mtc:
        mtc2[i] = p1

# mtc1 = [np.argmin(dist[i, :]) for i in range(dist.shape[0])]
# mtc2 = [np.argmin(dist[:, i]) for i in range(dist.shape[1])]

plt.close()
fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
ax[0].imshow(img1_gray, cmap='gray')
ax[1].imshow(img2_gray, cmap='gray')

good_pairs = []

for i, m in enumerate(mtc1):
    if mtc2[m] == i and m != -1:
        ax[0].scatter(fpts1[i, 1], fpts1[i, 0])
        ax[1].scatter(fpts2[m, 1], fpts2[m, 0])
        good_pairs.append((i, m))
plt.show()

# %%

canvas = np.concatenate([img1_gray, img2_gray], axis=1)
plt.figure(figsize=(20, 10))
plt.imshow(canvas, cmap='gray')

for pair in good_pairs:
    pp = np.hstack([fpts1[pair[0]], fpts2[pair[1]]])
    plt.plot([fpts1[pair[0]][1], fpts2[pair[1]][1] + img1.shape[1]],
             [fpts1[pair[0]][0], fpts2[pair[1]][0]], 'x-', linewidth=1, markersize=12)

plt.show()
