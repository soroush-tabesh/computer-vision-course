import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt

all_images = []
image_names = glob.glob('./data/hw2/checkerboard/*.jpg')
for image_name in sorted(image_names):
    all_images.append(plt.imread(image_name))


# %%
def get_camera_matrix(images, calibration_flags=None):
    board_size = (6, 9)

    pts2d = []
    pts3d = np.zeros((len(images), board_size[0] * board_size[1], 3), np.float32)
    pts3d[:, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * 0.022
    pts3d = list(pts3d)

    for img in images:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(img_gray, board_size,
                                                cv.CALIB_CB_ADAPTIVE_THRESH |
                                                cv.CALIB_CB_FAST_CHECK |
                                                cv.CALIB_CB_NORMALIZE_IMAGE)
        corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.0001))
        pts2d.append(corners2)

    ret, mtx, dis, rs, ts = cv.calibrateCamera(pts3d, pts2d, images[0].shape[:2][::-1], None, None,
                                               flags=calibration_flags)
    return mtx


mat1 = get_camera_matrix(all_images[0:10])
print(f'1: {mat1}')
mat2 = get_camera_matrix(all_images[5:15])
print(f'2: {mat2}')
mat3 = get_camera_matrix(all_images[10:20])
print(f'3: {mat3}')
mat4 = get_camera_matrix(all_images[0:20])
print(f'4: {mat4}')

# %%
mats = [mat1, mat2, mat3, mat4]
dists = np.zeros((len(mats), len(mats)))

for i, matA in enumerate(mats):
    for j, matB in enumerate(mats):
        dists[i, j] = np.linalg.norm(matA - matB) / np.linalg.norm(matA)

print(dists)

# %%
mat = get_camera_matrix(all_images[0:20],
                        calibration_flags=cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_FIX_PRINCIPAL_POINT)
print(f'Focal Distance = {mat[0, 0]:.2f}px')
