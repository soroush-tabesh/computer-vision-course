# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import math


def imshow(*srcs):
    plt.close()
    for src in srcs:
        plt.figure(figsize=(10, 10))
        t = (src - src.min()) / (src.max() - src.min())
        plt.imshow(t)
        plt.show()


# parameters

cams_h = 25.
cams_dist = 40.

cam1_f = 500  # camera #1 is the top view
cam2_f = 500  # camera #2 is the side camera
cam1_size = np.array([2000, 2000])  # xy system
cam2_size = np.array([256, 256])  # xy system

plane_normal = np.array([0, 0, -1])
d = 25.

# calc homography

c2 = np.array([0, cams_dist, 0])
r2 = Rotation.from_euler('x', math.atan(-cams_dist / cams_h)).as_matrix()
t2 = -np.matmul(r2, c2)

k1 = np.array([[cam1_f, 0, cam1_size[0] / 2],
               [0, cam1_f, cam1_size[1] / 2],
               [0, 0, 1]])
k2 = np.array([[cam2_f, 0, cam2_size[0] / 2],
               [0, cam2_f, cam2_size[1] / 2],
               [0, 0, 1]])

h = np.matmul(k2, r2 - np.matmul(t2[..., None], plane_normal[..., None].T) / d)
h = np.matmul(h, np.linalg.inv(k1))
h = np.linalg.inv(h)

# warping

img = plt.imread('./data/hw1/logo.png')
res = cv.warpPerspective(img, h, tuple(cam1_size))
plt.imshow(res)
plt.show()
