# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = plt.imread('./data/hw1/im03.jpg')
img2 = plt.imread('./data/hw1/im04.jpg')

# %%
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

img_corners = cv.drawMatches(img1, kp1, img2, kp2, None, None, singlePointColor=(0, 255, 0))
plt.imsave('./out/res13_corners.jpg', img_corners)

# %%
bf = cv.BFMatcher_create(cv.NORM_L2, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
# %%
img_correspondence = img_corners.copy()
for match in matches:
    cv.drawMatches(img1, [kp1[match.queryIdx]], img2, [kp2[match.trainIdx]], None, img_correspondence
                   , singlePointColor=(0, 0, 255), flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
plt.imsave('./out/res14_correspondence.jpg', img_correspondence)
# %%

img_matches = img_correspondence.copy()
cv.drawMatches(img1, kp1, img2, kp2, matches, img_matches
               , matchColor=(0, 0, 255),
               flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS + cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
plt.imsave('./out/res15_matches.jpg', img_matches)

# %%

img_matches_some = img_correspondence.copy()
cv.drawMatches(img1, kp1, img2, kp2, matches[:20], img_matches_some
               , matchColor=(0, 0, 255),
               flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS + cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
plt.imsave('./out/res16.jpg', img_matches_some)

# %%

img1_final_points = np.float32([kp1[m.queryIdx].pt for m in matches])
img2_final_points = np.float32([kp2[m.trainIdx].pt for m in matches])

M, mask = cv.findHomography(img2_final_points, img1_final_points, cv.RANSAC,
                            ransacReprojThreshold=3,
                            maxIters=500000,
                            confidence=0.995)

# %%

img_inliers = cv.drawMatches(img1, None, img2, None, None, None)
cv.drawMatches(img1, kp1, img2, kp2, matches, img_inliers
               , matchColor=(0, 0, 255), singlePointColor=(0, 0, 255),
               flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv.drawMatches(img1, kp1, img2, kp2, matches, img_inliers, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0),
               matchesMask=mask,
               flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
plt.imsave('./out/res17.jpg', img_inliers)

# %%

M_p, mask_p = cv.findHomography(img2_final_points, img1_final_points, cv.RANSAC,
                                ransacReprojThreshold=3,
                                maxIters=5000,
                                confidence=0.995)
img_mismatch = cv.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0),
                              matchesMask=mask_p,
                              flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
plt.imsave('./out/res18_mismatch.jpg', img_mismatch)

# %%
img2_warped = cv.warpPerspective(img2, M, img1.shape[:2][::-1])
plt.imsave('./out/res19.jpg', img2_warped)
