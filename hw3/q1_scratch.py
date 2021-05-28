import cv2 as cv
import time
import numpy as np
from matplotlib import pyplot as plt


def imshow(*srcs, bgr=False, explicit=True):
    if explicit:
        plt.close()
    fig, ax = plt.subplots(ncols=len(srcs), figsize=(len(srcs) * 10, 10))
    for i, src in enumerate(srcs):
        if len(src.shape) == 3 and bgr:
            src = src[..., ::-1]
        t = (src - src.min()) / max(src.max() - src.min(), 1e-6)
        if (len(srcs)) > 1:
            ax[i].imshow(t, cmap='gray')
        else:
            ax.imshow(t, cmap='gray')
    if explicit:
        plt.show()


img_o = plt.imread('./data/hw3/vns.jpg')

imshow(img_o)

lines_x = np.array([[3542, 969, 3956, 1087],
                    [3604, 1283, 4014, 1376],
                    [1599, 692, 1991, 787],
                    [1248, 203, 1653, 322],
                    [1767, 2665, 2146, 2662],
                    [1654, 2493, 2049, 2500],
                    [3619, 2557, 4125, 2559],
                    [3692, 175, 3971, 295],
                    [551, 1369, 822, 1406],
                    [810, 1535, 1042, 1565]])
lines_y = np.array([[3005, 1025, 3544, 971],
                    [2062, 1247, 2827, 1171],
                    [3007, 260, 3691, 175],
                    [2987, 461, 3653, 384],
                    [2579, 1990, 2149, 2019],
                    [3621, 2580, 2961, 2612],
                    [701, 1057, 387, 1092],
                    [1161, 201, 702, 264],
                    [1492, 749, 976, 809],
                    [2600, 2462, 2168, 2487]])
lines_z = np.array([[2099, 2057, 2068, 1259],
                    [2042, 2058, 2004, 1105],
                    [3613, 2545, 3550, 1246],
                    [4100, 2294, 4003, 522],
                    [702, 1059, 681, 282],
                    [1654, 2494, 1577, 705],
                    [2992, 1334, 2952, 471]])

# Automatic line detection
# %% preparation

img = img_o.copy()
img = cv.GaussianBlur(img, (0, 0), 9)
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
edges = cv.Canny(img, 10, 30, L2gradient=True)
imshow(edges)

# %%
lines = cv.HoughLines(edges, 3, np.pi / 180, 460)
lines = np.array(sorted(lines, key=lambda x: x[0, 1])[:])
frame = np.zeros_like(img_o) + 255
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * a)
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * a)
    cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
plt.imshow(frame)
for line in lines_x:
    plt.plot(line[::2], line[1:][::2], c='g')
for line in lines_y:
    plt.plot(line[::2], line[1:][::2], c='r')
for line in lines_z:
    plt.plot(line[::2], line[1:][::2], c='b')
plt.show()
# plt.scatter(lines[:, 0, 0], lines[:, 0, 1], s=0.2)
# plt.show()

# %%
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100)
frame = np.zeros_like(img_o) + 255
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
imshow(frame)

# %%
lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100)
frame = np.zeros_like(img_o) + 255
for line in lines:
    x1, y1, x2, y2 = line[0]
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
    else:
        m = 10000 * np.sign(y2 - y1)
    cv.line(frame, (x1 - 1000, int(y1 - 1000 * m)), (x1 + 1000, int(y2 + 1000 * m)), (255, 0, 0), 3)
imshow(frame)


# %% manual line detection
def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


def get_input_lines(img, title='', min_lines=2):
    plt.clf()
    plt.imshow(img)
    tellme(title + ' press any key to start')
    plt.waitforbuttonpress()
    while True:
        lines = []
        init = []
        while len(init) // 2 < min_lines:
            tellme(f'Draw at least {min_lines} lines, press \'esc\' to continue.')
            init = np.asarray(plt.ginput(-1, timeout=-1))
            if len(init) // 2 < min_lines:
                tellme('Too few lines, starting over')
                time.sleep(1)  # Wait a second
        ph = []
        for t in range(0, len(init) - 1, 2):
            lines.append([*init[t], *init[t + 1]])
            p = plt.plot(init[t:t + 2, 0], init[t:t + 2, 1], 'r', lw=1)
            ph.append(p)
        tellme('Happy? \'esc\' for yes, mouse click for no')
        if plt.waitforbuttonpress():
            break
        for p in ph:
            for pt in p:
                pt.remove()
    return np.round(np.array(lines)).astype(np.int)


# lines_x = get_input_lines(img_o, 'x axis')
# lines_y = get_input_lines(img_o, 'y axis')
# lines_z = get_input_lines(img_o, 'z axis')
# print(lines_x)
# print(lines_y)
# print(lines_z)


# %%

def find_intersection(lines):
    A = np.zeros((len(lines), 2))
    b = np.zeros(len(lines))
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i]
        if x1 == x2:
            A[i, 0] = 1
            A[i, 1] = 0
            b[i] = x1
        else:
            A[i, 0] = (y2 - y1) / (x2 - x1)
            A[i, 1] = -1
            b[i] = np.dot(A[i, :], lines[i, :2])
    return np.append(np.linalg.lstsq(A, b, rcond=None)[0], [1])


plt.imshow(img_o)

vx = find_intersection(lines_x)
vy = find_intersection(lines_y)
vz = find_intersection(lines_z)

plt.scatter(vx[0], vx[1], s=1)
plt.scatter(vy[0], vy[1], s=1)
# plt.scatter(vz[0], vz[1], s=1)
plt.plot([vx[0], vy[0]], [vx[1], vy[1]], linewidth=1, marker='+')

# for line in lines_z:
#     # plt.plot(line[::2], line[1:][::2], c='g')
#     x1, y1, x2, y2 = line
#     if x2 - x1 != 0:
#         m = (y2 - y1) / (x2 - x1)
#     else:
#         m = 10000 * np.sign(y2 - y1)
#     plt.plot((x1 - 10000, x1 + 10000), (int(y1 - 10000 * m), int(y2 + 10000 * m)), c='g')

plt.show()


# %%
def find_focal_principal(vpts):
    A = np.array([[vpts[0][0] - vpts[2][0], vpts[0][1] - vpts[2][1]],
                  [vpts[1][0] - vpts[2][0], vpts[1][1] - vpts[2][1]]])
    b = np.array([vpts[1][0] * (vpts[0][0] - vpts[2][0]) + vpts[1][1] * (vpts[0][1] - vpts[2][1]),
                  vpts[0][0] * (vpts[1][0] - vpts[2][0]) + vpts[0][1] * (vpts[1][1] - vpts[2][1])])
    p = np.linalg.solve(A, b)
    f2 = -p[0] ** 2 - p[1] ** 2 + (vpts[0][0] + vpts[1][0]) * p[0] + (vpts[0][1] + vpts[1][1]) * p[1] - (
            vpts[0][0] * vpts[1][0] + vpts[0][1] * vpts[1][1])
    return p, np.sqrt(f2)


p, f = find_focal_principal([vx, vy, vz])
K = np.array([[f, 0, p[0]],
              [0, f, p[1]],
              [0, 0, 1]])
Ki = np.linalg.inv(K)
