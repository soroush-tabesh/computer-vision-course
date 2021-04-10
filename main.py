import cv2 as cv
import numpy as np
import time

r = (0, 0, 255)
g = (0, 255, 0)
b = (255, 0, 0)


def save_2_image(im1, im2, name):
    h1, w1, c = im1.shape
    h2, w2 = im2.shape[:2]
    result = np.empty((max(h1, h2), w1 + w2, c), 'uint8')
    result[:h1, 0:w1, :] = im1
    result[:h2, w1:w2 + w1, :] = im2
    cv.imwrite(name, result)


def main():
    im1 = cv.imread('./data/hw1/im03.jpg')
    im2 = cv.imread('./data/hw1/im04.jpg')
    gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    # sift = cv.SIFT_create(contrastThreshold=0.07, edgeThreshold=2, sigma=1)
    sift = cv.SIFT_create()
    key_points1, descriptor1 = sift.detectAndCompute(im1, None)
    key_points2, descriptor2 = sift.detectAndCompute(im2, None)
    corners1 = cv.drawKeypoints(im1, key_points1, None, color=g)
    corners2 = cv.drawKeypoints(im2, key_points2, None, color=g)
    save_2_image(corners1, corners2, "res13_corners.jpg")
    matcher = cv.BFMatcher_create(cv.NORM_L2, crossCheck=True)
    matches = matcher.match(descriptor1, descriptor2)
    matches = sorted(matches, key=lambda ma: ma.distance)
    matched_kp_1 = []
    matched_kp_2 = []
    for m in matches:
        matched_kp_1.append(key_points1[m.queryIdx])
        matched_kp_2.append(key_points2[m.trainIdx])
    print("key points 1 =", len(key_points1))
    print("key points 2 =", len(key_points2))
    print("matched points =", len(matches))
    correspondences1 = cv.drawKeypoints(corners1, matched_kp_1, None, color=b)
    correspondences2 = cv.drawKeypoints(corners2, matched_kp_2, None, color=b)
    save_2_image(correspondences1, correspondences2, "res14_correspondences.jpg")
    res15 = cv.drawMatches(im1, key_points1, im2, key_points2, matches, None, matchColor=b, singlePointColor=g)
    cv.imwrite("res15_matches.jpg", res15)
    res16 = cv.drawMatches(im1, key_points1, im2, key_points2, matches[:20], None, matchColor=b, singlePointColor=g)
    cv.imwrite("res16.jpg", res16)
    src_pts = np.float32([key_points2[m.trainIdx].pt for m in matches])
    dst_pts = np.float32([key_points1[m.queryIdx].pt for m in matches])
    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0, maxIters=200000)
    res17_1 = cv.drawKeypoints(im1, matched_kp_1, None, color=r)
    res17_2 = cv.drawKeypoints(im2, matched_kp_2, None, color=r)
    res17 = cv.drawMatches(res17_1, key_points1, res17_2, key_points2, matches, None, matchColor=b, matchesMask=mask,
                           flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("res17.jpg", res17)
    h1, w1 = im1.shape[:2]
    inlier = []
    for i in np.where(mask == 1)[0]:
        inlier.append(matches[i])
    src_pts = np.float32([key_points2[m.trainIdx].pt for m in inlier]).reshape(-1, 1, 2)
    dst_pts = cv.perspectiveTransform(src_pts, homography)
    bad_inlier = []
    for i, m in enumerate(inlier):
        x, y = dst_pts[i, 0]
        if x < 0 or x >= w1 or y < 0 or y >= h1:
            bad_inlier.append(m)
    if len(bad_inlier) != 0:
        res18 = cv.drawMatches(im1, key_points1, im2, key_points2, bad_inlier, None, matchColor=r,
                               flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite("res18.jpg", res18)
    warped_image = cv.warpPerspective(im2, homography, (w1, h1))
    cv.imwrite("res19.jpg", warped_image)


if __name__ == '__main__':
    start = time.time()
    main()
    print("execution time =", time.time() - start)
