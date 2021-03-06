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


img1 = plt.imread('./data/hw1/im03.jpg')
img2 = plt.imread('./data/hw1/im04.jpg')

img1_c = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
img2_c = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

imshow(img1, img2)

# %%
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

imshow(cv.drawKeypoints(img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
       cv.drawKeypoints(img2, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

# %%
# bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=True)
#
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
match_ratio_threshold = 0.7
matches = cv.BFMatcher().knnMatch(des1, des2, k=2)
matches = [m1 for m1, m2 in matches if m1.distance < match_ratio_threshold * m2.distance]
# %%

img1_final_points = np.float32([kp1[m.queryIdx].pt for m in matches])
img2_final_points = np.float32([kp2[m.trainIdx].pt for m in matches])

M, mask = cv.findHomography(img2_final_points, img1_final_points, cv.RANSAC,
                            ransacReprojThreshold=3,
                            maxIters=500000,
                            confidence=0.995)
img4 = cv.warpPerspective(img2, M, img1.shape[:2][::-1])

imshow(img1, img4)
# %%
img3_3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                        matchColor=(0, 255, 0),  # draw matches in green color
                        # singlePointColor=(0, 0, 255),
                        flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
imshow(img3_3)
# %%
img3_1 = cv.drawMatches(img1, kp1, img2, kp2, matches, None,
                        matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=(0, 0, 255),
                        matchesMask=(mask),  # draw only inliers
                        flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
img3_2 = cv.drawMatches(img1, kp1, img2, kp2, matches, None,
                        matchColor=(255, 0, 0),  # draw matches in green color
                        singlePointColor=(0, 0, 255),
                        matchesMask=(1 - mask),  # draw only inliers
                        flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
# imshow((img3_1.astype(np.int) + img3_2.astype(np.int)) // 2)
# imshow(img3_1)
# %%
plt.imshow(img1, extent=(10, 20, 30, 40))
plt.show()
# for match, flag in zip(matches, mask):
#     plt
