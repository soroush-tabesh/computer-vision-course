import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

all_images = []
image_names = glob.glob('./data/hw2/checkerboard/*.jpg')
for image_name in sorted(image_names):
    all_images.append(plt.imread(image_name))


# %%
def get_camera_matrix(images):
    board_size = (6, 9)

    pts2d = []
    pts3d = np.zeros((len(images), board_size[0] * board_size[1], 3), np.float32)
    pts3d[:, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * 0.022
    pts3d = list(pts3d)

    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img_gray, board_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK +
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)
        corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        pts2d.append(corners2)

    _, mtx, _, _, _ = cv2.calibrateCamera(pts3d, pts2d, images[0].shape[:2][::-1], None, None)
    return mtx


print(get_camera_matrix(all_images[0:10]))
print(get_camera_matrix(all_images[5:15]))
print(get_camera_matrix(all_images[10:20]))
print(get_camera_matrix(all_images[0:20]))
