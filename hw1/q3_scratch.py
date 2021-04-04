# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, util


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


img1 = plt.imread('./data/hw1/im03.jpg')
img2 = plt.imread('./data/hw1/im04.jpg')

img1_c = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
img2_c = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

imshow(img1, img2)

# %%
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1_c, None)
kp2, des2 = sift.detectAndCompute(img2_c, None)

imshow(cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
       cv.drawKeypoints(img2, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

# %%
bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# %%
nn = 1000

# img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:], None, flags=2)
# imshow(img3)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:]])[:, ::-1]
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:]])[:, ::-1]

M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, ransacReprojThreshold=3, maxIters=2000, confidence=0.995)
img4 = cv.warpPerspective(img2, M, img1.shape[:2][::-1])

imshow(img4, img1)

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=mask,  # draw only inliers
                   flags=2)

img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
imshow(img3)
# %%
matches = [matches[i] for i in range(len(matches)) if mask[i, 0] > 0]
