# %%
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pickle
from skimage import util, color, draw
import time

cap = cv.VideoCapture('./data/hw2/video.mp4')


def get_frame(video_cap, f_num):
    video_cap.set(cv.CAP_PROP_POS_FRAMES, f_num)
    return video_cap.read()[1]


def imshow(*srcs, bgr=True):
    plt.close()
    fig, ax = plt.subplots(ncols=len(srcs), figsize=(len(srcs) * 10, 10))
    for i, src in enumerate(srcs):
        if len(src.shape) == 3 and bgr:
            src = src[..., ::-1]
        t = (src - src.min()) / (src.max() - src.min())
        if (len(srcs)) > 1:
            ax[i].imshow(t, cmap='gray')
        else:
            ax.imshow(t, cmap='gray')
    plt.show()


imshow(get_frame(cap, 270)[..., ::-1], get_frame(cap, 450)[..., ::-1])


# %% part 1
def find_homography(img1, img2, match_ratio_threshold=0.8, surf_hessian=400, gpu=False, down_scale=0.5):
    if gpu:
        r1 = time.time()
        gpu_img1 = cv.cuda_GpuMat(img1)
        gpu_img2 = cv.cuda_GpuMat(img2)

        gpu_img1 = cv.cuda.resize(gpu_img1, (0, 0), gpu_img1, down_scale, down_scale, interpolation=cv.INTER_AREA)
        gpu_img2 = cv.cuda.resize(gpu_img2, (0, 0), gpu_img2, down_scale, down_scale, interpolation=cv.INTER_AREA)

        gpu_img1 = cv.cuda.cvtColor(gpu_img1, cv.COLOR_BGR2GRAY)
        gpu_img2 = cv.cuda.cvtColor(gpu_img2, cv.COLOR_BGR2GRAY)

        surf_gpu = cv.cuda.SURF_CUDA_create(surf_hessian)
        matcher = cv.cuda.DescriptorMatcher_createBFMatcher(cv.NORM_L2)

        kp1_gpu, des1 = surf_gpu.detectWithDescriptors(gpu_img1, None)
        kp2_gpu, des2 = surf_gpu.detectWithDescriptors(gpu_img2, None)

        matches = matcher.knnMatch(des1, des2, k=2)

        kp1 = cv.cuda_SURF_CUDA.downloadKeypoints(surf_gpu, kp1_gpu)
        kp2 = cv.cuda_SURF_CUDA.downloadKeypoints(surf_gpu, kp2_gpu)

        matches = [m1 for m1, m2 in matches if m1.distance < match_ratio_threshold * m2.distance]
        r2 = time.time()
    else:
        algo = cv.ORB_create()

        img1 = cv.resize(img1, (0, 0), img1, down_scale, down_scale, interpolation=cv.INTER_AREA)
        img2 = cv.resize(img2, (0, 0), img2, down_scale, down_scale, interpolation=cv.INTER_AREA)

        kp1, des1 = algo.detectAndCompute(img1, None)
        kp2, des2 = algo.detectAndCompute(img2, None)

        matches = cv.BFMatcher(cv.NORM_L2).knnMatch(des1, des2, k=2)
        matches = [m1 for m1, m2 in matches if m1.distance < match_ratio_threshold * m2.distance]

    img1_final_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    img2_final_points = np.float32([kp2[m.trainIdx].pt for m in matches])

    M, mask = cv.findHomography(img1_final_points, img2_final_points, cv.RANSAC,
                                ransacReprojThreshold=3,
                                maxIters=500000,
                                confidence=0.995)
    r3 = time.time()
    cnv = np.array([[down_scale, 0, 0], [0, down_scale, 0], [0, 0, 1]])
    cnvi = np.array([[1 / down_scale, 0, 0], [0, 1 / down_scale, 0], [0, 0, 1]])
    # img3_1 = cv.drawMatches(img1, kp1, img2, kp2, matches, None,
    #                         matchColor=(0, 255, 0),  # draw matches in green color
    #                         singlePointColor=(0, 0, 255),
    #                         matchesMask=mask,  # draw only inliers
    #                         flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    # imshow(img3_1[..., ::-1])
    # print(r2-r1,r3-r2)
    return np.matmul(cnvi, np.matmul(M, cnv))


# cap = cv.VideoCapture(video_path)
# find_homography(get_frame(cap, 270), get_frame(cap, 450), gpu=True)
# cap.release()

# %% load all frames (don't run it locally!)
all_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    all_frames.append(frame)

# %% and find homography for all!
homo_to_450 = np.zeros((900, 3, 3))
homo_to_450[450] = np.eye(3)

key_frames_dist = 180

for i in range(451, 900, 1):
    stp = (i - 450 - 1) % key_frames_dist + 1
    print(i, stp, i - stp)
    homo_to_450[i] = np.matmul(find_homography(all_frames[i], all_frames[i - stp], gpu=True), homo_to_450[i - stp])

for i in range(449, -1, -1):
    stp = (450 - i - 1) % key_frames_dist + 1
    print(i, stp, i + stp)
    homo_to_450[i] = np.matmul(find_homography(all_frames[i], all_frames[i + stp], gpu=True), homo_to_450[i + stp])

import pickle

with open('homo_to_450_surf_gpu.pickle', 'wb') as file:
    pickle.dump(homo_to_450, file)

# %%
with open('./hw2/homo_to_450_surf_gpu.pickle', 'rb') as file:
    homo_to_450 = pickle.load(file)

# %% task 1_1
frame450_rect = all_frames[450].copy()
frame270_rect = all_frames[270].copy()
rect_pts = np.array([[500, 500], [1000, 500], [1000, 1000], [500, 1000]], dtype=np.int32).reshape((-1, 1, 2))
cv.polylines(frame450_rect, [rect_pts], True, (0, 0, 255), thickness=5)
rect_pts = cv.perspectiveTransform(rect_pts.astype(np.float32), np.linalg.inv(homo_to_450[270])).astype(np.int32)
cv.polylines(frame270_rect, [rect_pts], True, (0, 0, 255), thickness=5)
imshow(frame270_rect, frame450_rect)


# cv.imwrite('./drive/MyDrive/res01-450-rect.jpg',frame450_rect)
# cv.imwrite('./drive/MyDrive/res02-270-rect.jpg',frame270_rect)


# %% task 1_2
def warp_perspective(src, mat, sz, gpu=False, flt=True):
    res = None
    if gpu:
        src = util.img_as_ubyte(src)
        gpu_imgt = cv.cuda_GpuMat(src)
        tmp = cv.cuda.warpPerspective(gpu_imgt, mat, sz)
        res = tmp.download()
    else:
        res = cv.warpPerspective(src, mat, sz)
    if flt:
        return util.img_as_float64(res)
    else:
        return res


def find_bounding_frame(mats, frame_size):  # frame size = (w,h)
    predicted_points = []
    w, h = frame_size
    src_points = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    for mat in mats:
        t = cv.perspectiveTransform(src_points.reshape(-1, 1, 2), mat).reshape(src_points.shape)
        predicted_points.append(t)
    predicted_points = np.concatenate(tuple(predicted_points))
    pts_min = np.min(predicted_points, axis=0)
    pts_max = np.max(predicted_points, axis=0)
    pts_diff = np.int64(pts_max - pts_min)
    mat_t = np.array([[1, 0, -pts_min[0]], [0, 1, -pts_min[1]], [0, 0, 1]])
    return tuple(pts_diff), mat_t


h, w = all_frames[0].shape[:2]

p_diff, M_t = find_bounding_frame(homo_to_450[[450, 270]], (w, h))
M_p = np.matmul(M_t, homo_to_450[270])

df_mask = np.ones_like(all_frames[450]) * 255

f270_warped = warp_perspective(all_frames[270], M_p, tuple(p_diff))
f450_warped = warp_perspective(all_frames[450], M_t, tuple(p_diff))
f450_mask_warped = warp_perspective(df_mask, M_t, tuple(p_diff)) > 0.8

tmp3 = f270_warped.copy()
tmp3[f450_mask_warped] = f450_warped[f450_mask_warped]

imshow(f270_warped, f450_warped, tmp3)


# cv.imwrite('./drive/MyDrive/res03-270-450-panorama.jpg', tmp3)


# %% task 2
def multi_band_blending(img1, img2, mask,
                        iterations=5,
                        m_ratio=0.8,
                        bandwidth_low=10,
                        bandwidth_high=20,
                        m_cutoff=12):
    def pyr_up(src, cutoff, ratio):
        lowpassed = cv.GaussianBlur(src, (2 * cutoff + 1, 2 * cutoff + 1), 0,
                                    borderType=cv.BORDER_REFLECT101)
        src -= lowpassed
        return cv.resize(lowpassed, (0, 0), None, ratio, ratio, cv.INTER_AREA)

    def blend(src, tar, mask, bandwidth):
        mask = cv.GaussianBlur(mask, (2 * bandwidth + 1, 2 * bandwidth + 1), 0
                               , borderType=cv.BORDER_REFLECT101)[:, :, None]
        return src * mask + tar * (1 - mask)

    if len(mask.shape) == 3:
        mask = color.rgb2gray(mask)

    pyr_lap_1 = [util.img_as_float64(img1, force_copy=True)]
    pyr_lap_2 = [util.img_as_float64(img2, force_copy=True)]
    pyr_mask = [util.img_as_float64(mask, force_copy=True)]

    for i in range(iterations):
        pyr_lap_1.append(pyr_up(pyr_lap_1[i], m_cutoff, m_ratio))
        pyr_lap_2.append(pyr_up(pyr_lap_2[i], m_cutoff, m_ratio))
        pyr_mask.append(cv.resize(pyr_mask[i], (0, 0), None, m_ratio, m_ratio, cv.INTER_NEAREST))

    pyr_lap_1[iterations] = blend(pyr_lap_1[iterations], pyr_lap_2[iterations], pyr_mask[iterations], bandwidth_low)

    for i in range(iterations - 1, -1, -1):
        pyr_lap_1[i] = blend(pyr_lap_1[i], pyr_lap_2[i], pyr_mask[i], bandwidth_high)
        tmp = cv.resize(pyr_lap_1[i + 1], pyr_lap_1[i].shape[:2][::-1], interpolation=cv.INTER_AREA)
        pyr_lap_1[i] += tmp

    res = np.clip(pyr_lap_1[0], 0, 1)
    return res


def get_mag(src):
    src = src ** 2
    res = np.zeros(src.shape[:2])
    if len(src.shape) > 2:
        for i in range(3):
            res += src[:, :, i]
        res /= 3
    else:
        res += src
    res = np.sqrt(res)
    return res


def find_shortest_path(mat, spr=1, diag=1):
    dp = np.ones_like(mat) * 1000000000
    dp_arg = np.ones(mat.shape, dtype=int) * -1
    dp[0, :] = 0
    dp_arg[0, :] = -1
    for i in range(1, mat.shape[0]):
        print('\b' * 20, i)
        for j in range(mat.shape[1]):
            for j2 in range(max(0, j - spr), min(mat.shape[1], j + spr + 1)):
                dist = 0
                if abs(j - j2) > diag:
                    pass
                else:
                    dist = dp[i - 1, j2] + mat[i - 1, j2]
                if dist < dp[i, j]:
                    dp[i, j] = dist
                    dp_arg[i, j] = j2
    arg_mn = np.argmin(dp[-1, :])
    arg_i = mat.shape[0] - 1
    res = [(arg_i, arg_mn)]
    while arg_i > 0:
        arg_mn = dp_arg[arg_i, arg_mn]
        arg_i -= 1
        res.append((arg_i, arg_mn))
    res.reverse()
    return np.array(res)


def get_shortest_path_mask(mat, spr=1, diag=1, ker_prep=5):
    print(mat.shape)
    mat = cv.GaussianBlur(mat, (ker_prep, ker_prep), 0)
    verts = find_shortest_path(mat, spr, diag)
    verts = np.concatenate((verts, [[mat.shape[0], 0], [0, 0]]), axis=0)
    return draw.polygon2mask(mat.shape[:2], verts)


def get_overlap_mask(img1, img2):
    img1_mask = 1.0 * (img1 > 0.001)
    img2_mask = 1.0 * (img2 > 0.001)

    kernel = np.ones((25, 25), np.uint8)
    img1_mask = cv.morphologyEx(img1_mask, cv.MORPH_CLOSE, kernel, iterations=1)
    img2_mask = cv.morphologyEx(img2_mask, cv.MORPH_CLOSE, kernel, iterations=1)

    overlap = img1_mask * img2_mask
    return overlap, img1_mask, img2_mask


def find_optimal_mask(img1, img2):
    img1 = util.img_as_float64(img1, force_copy=True)
    img2 = util.img_as_float64(img2, force_copy=True)

    overlap, img1_mask, img2_mask = get_overlap_mask(img1, img2)

    ag = np.argwhere(overlap)
    bound_min = np.min(ag, axis=0)
    bound_max = np.max(ag, axis=0)
    conf_dist = (bound_max[1] - bound_min[1]) // 20

    diff_mag = get_mag(img1 - img2)
    diff_mag[:, bound_min[1]:bound_min[1] + conf_dist] = 1
    diff_mag[:, bound_max[1] - conf_dist:bound_max[1]] = 1

    msk_ovr = 1 - 1.0 * get_shortest_path_mask(diff_mag[:, bound_min[1]:bound_max[1]])
    msk = 1.0 * img2_mask
    msk[:, bound_min[1]:bound_max[1]] *= msk_ovr[..., None]

    return msk


h, w = all_frames[0].shape[:2]
key_frame_idx = [90, 270, 450, 630, 810]
p_diff, M_t = find_bounding_frame(homo_to_450[key_frame_idx], (w, h))

res04 = warp_perspective(all_frames[key_frame_idx[0]], np.matmul(M_t, homo_to_450[key_frame_idx[0]]), p_diff)
# imshow(res04)

for idx in key_frame_idx[1:]:
    tmp = warp_perspective(all_frames[idx], np.matmul(M_t, homo_to_450[idx]), p_diff)
    mask_m = find_optimal_mask(res04, tmp)
    mask_overlap = get_overlap_mask(res04, tmp)[0]
    res04 = multi_band_blending(tmp, res04, mask_m) * mask_overlap + (res04 + tmp) * (1 - mask_overlap)
    # imshow(res04, tmp, mask_m)
    # imshow(res04)

imshow(res04)

# %% task 3
full_dim, M_t_full = find_bounding_frame(homo_to_450, (w, h))
vw = cv.VideoWriter('./drive/MyDrive/res05-reference-plane.mp4', cv.VideoWriter_fourcc(*'MP4V'), 30, full_dim)
for i, (frame, mat) in enumerate(zip(all_frames, homo_to_450)):
    vw.write(warp_perspective(frame, np.matmul(M_t_full, mat), full_dim, gpu=True))
    print('\b\b\b\b\b', i)
vw.release()


# %% task 4

def get_frame_corners(mat, sz):  # sz = (w,h)
    src_points = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    dst_points = cv.perspectiveTransform(src_points.reshape(-1, 1, 2), mat).reshape(src_points.shape)
    return dst_points


strip_width = 300
full_dim, M_t_full = find_bounding_frame(homo_to_450, (w, h))
res06 = np.zeros((full_dim[1], full_dim[0], 3), dtype=np.uint8)

for t in range((full_dim[0] + strip_width - 1) // strip_width):
    x_min = strip_width * t
    x_max = min(strip_width * (t + 1), full_dim[0])
    print(t)
    fr_idx = []
    for i, mat in enumerate(homo_to_450):
        corners = get_frame_corners(np.matmul(M_t_full, mat), (w, h))
        if corners[0, 0] <= x_min + 200 and corners[1, 0] <= x_min + 200 and corners[2, 0] >= x_max - 200 and corners[
            3, 0] >= x_max - 200:
            fr_idx.append(i)

    fr_stack = np.zeros((len(fr_idx), full_dim[1], x_max - x_min, 3), dtype=np.uint8)

    for i, idx in enumerate(fr_idx):
        mat = homo_to_450[idx]
        frame = all_frames[idx]
        tr = warp_perspective(frame,
                              np.matmul(np.array([[1, 0, -x_min], [0, 1, 0], [0, 0, 1]]), np.matmul(M_t_full, mat)),
                              (x_max - x_min, full_dim[1]), flt=False)
        # print(tr.shape,fr_stack.shape)
        fr_stack[i, ...] = tr
        del tr

    med = np.median(fr_stack, axis=0)
    # imshow(med)
    res06[:, x_min:x_max, :] = med
    del fr_stack
