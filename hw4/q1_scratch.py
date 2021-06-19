import os
import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature
from sklearn import utils, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

img = plt.imread('./data/hw4/pos/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
img = cv.resize(img, (256, 256), interpolation=cv.INTER_AREA)

# %%
t1 = time.time()
for t in range(100):
    cv.imread('./data/hw4/pos/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
print(time.time() - t1)

# %%
t1 = time.time()
for t in range(100):
    tt = feature.hog(img,
                     cells_per_block=(2, 2),
                     pixels_per_cell=(8, 8),
                     multichannel=True,
                     feature_vector=True).ravel()
print(time.time() - t1)
tt.shape

# %%
cell_size = (8, 8)
block_size = (2, 2)
n_bins = 9

t1 = time.time()
for t in range(100):
    hog = cv.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                     img.shape[0] // cell_size[0] * cell_size[0]),
                           _blockSize=(block_size[1] * cell_size[1],
                                       block_size[0] * cell_size[0]),
                           _blockStride=(cell_size[1], cell_size[0]),
                           _cellSize=(cell_size[1], cell_size[0]),
                           _nbins=n_bins)
    tt = hog.compute(img).ravel()
print(time.time() - t1)


# %%
def get_all_files(path, pat='jpg'):
    list_of_files = []
    for (dir_path, dir_names, filenames) in os.walk(path):
        list_of_files += [os.path.join(dir_path, file) for file in filenames
                          if file.split('.')[-1].lower() == pat.lower()]
    return list_of_files


# %%
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)


# %% #######################################################
def get_descriptor(src, cell_size=(8, 8), block_size=(2, 2), n_bins=9):
    hog = cv.HOGDescriptor(_winSize=(src.shape[1] // cell_size[1] * cell_size[1],
                                     src.shape[0] // cell_size[0] * cell_size[0]),
                           _blockSize=(block_size[1] * cell_size[1],
                                       block_size[0] * cell_size[0]),
                           _blockStride=(cell_size[1], cell_size[0]),
                           _cellSize=(cell_size[1], cell_size[0]),
                           _nbins=n_bins)
    return hog.compute(src).ravel()


def get_image(filename, size=(128, 128), margin=(50, 50, 50, 50), min_size=(150, 150)):
    src = cv.imread(filename)
    if src.shape[0] > min_size[0] and src.shape[1] > min_size[1]:
        src = src[margin[0]:-margin[1], margin[2]:-margin[3], :]
    return cv.resize(src, size, interpolation=cv.INTER_AREA)


# load data
# ctg_cnt = 12000
X_file = []
tmp = utils.shuffle(get_all_files('./data/hw4/pos'), random_state=0)  # [:ctg_cnt]
X_file += tmp
X_file += utils.shuffle(get_all_files('./data/hw4/neg'), random_state=0)  # [:ctg_cnt]
y = np.zeros(len(X_file), dtype=np.int32)
y[:len(tmp)] = 1

# split data
(X_train_file, X_test_file,
 y_train, y_test) = train_test_split(X_file, y,
                                     stratify=y, test_size=2000, random_state=0, shuffle=True)
(X_train_file, X_valid_file,
 y_train, y_valid) = train_test_split(X_train_file, y_train,
                                      stratify=y_train, test_size=2000, random_state=0, shuffle=True)

# # %% fit a scaler
# scaler = StandardScaler(with_mean=True, with_std=True)
# for i, filename in enumerate(X_train_file):
#     print('\b' * 20, i, end='')
#     scaler.partial_fit([get_descriptor(get_image(filename))])

# %% fit a gd linear svm
sgd = linear_model.SGDClassifier(loss='hinge',
                                 penalty='l2',
                                 fit_intercept=True,
                                 alpha=0.0001,
                                 shuffle=False,
                                 learning_rate='optimal')
for i, (filename, label) in enumerate(zip(X_train_file, y_train)):
    print('\b' * 20, i, end='')
    ft = get_descriptor(get_image(filename))
    # ft = scaler.transform([ft])[0]
    sgd.partial_fit([ft], [label], classes=[0, 1])


# %% evaluate model
def evaluate_model(clf, X_file, y):
    y_pred = np.zeros_like(y)
    y_score = np.zeros_like(y, dtype=float)
    for i, filename in enumerate(X_file):
        print('\b' * 20, i, end='')
        ft = get_descriptor(get_image(filename))
        y_pred[i] = clf.predict([ft])
        y_score[i] = clf.decision_function([ft])
    print('\b' * 20)
    return y_pred, y_score


def calculate_accuracy(y, y_pred, y_score):
    return metrics.accuracy_score(y, y_pred)


def calculate_roc(y, y_pred, y_score):
    return metrics.roc_curve(y, y_score)


def calculate_roc_auc(y, y_pred, y_score):
    return metrics.roc_auc_score(y, y_score)


def calculate_ap(y, y_pred, y_score):
    return metrics.average_precision_score(y, y_score)


def calculate_prc(y, y_pred, y_score):
    return metrics.precision_recall_curve(y, y_score)


# %%
eval_valid = evaluate_model(sgd, X_valid_file, y_valid)
print(f'validation accuracy = {calculate_accuracy(y_valid, *eval_valid)}')
sgd_threshold = eval_valid[1][eval_valid[1] > 0].mean() - 2 * eval_valid[1][eval_valid[1] > 0].std()
# %%
eval_test = evaluate_model(sgd, X_test_file, y_test)
score_test = calculate_accuracy(y_test, *eval_test)
roc_test = calculate_roc(y_test, *eval_test)
roc_auc_test = calculate_roc_auc(y_test, *eval_test)
prc_test = calculate_prc(y_test, *eval_test)
ap_test = calculate_ap(y_test, *eval_test)

# %%
plt.plot(roc_test[0], roc_test[1])
plt.title(f'ROC, AUC={roc_auc_test:.5f}')
plt.savefig('./hw4/assignment/out/res1.jpg')
plt.clf()

plt.plot(prc_test[0], prc_test[1])
plt.title(f'Precision-Recall Curve, AP={ap_test:.5f}')
plt.savefig('./hw4/assignment/out/res2.jpg')
plt.clf()


# %%
def FaceDetection(clf, img, window_size=(128, 128),
                  stride=15, scale_step=0.1, scale_range=(0.5, 2), padding=0.2, threshold=30):
    frame = np.zeros((int(img.shape[0] * (1 + padding)), int(img.shape[1] * (1 + padding)), 3), dtype=np.uint8)
    frame[int(img.shape[0] * padding / 2):int(img.shape[0] * padding / 2) + img.shape[0],
    int(img.shape[1] * padding / 2):int(img.shape[1] * padding / 2) + img.shape[1], :] = img

    frame_res = frame.copy()
    frame_scores = np.zeros(frame_res.shape[:2])
    frame_pos = np.zeros(frame_res.shape[:2] + (2,), dtype=int)

    for scale in np.arange(scale_range[0], scale_range[1], scale_step):
        print('\b' * 20, scale, end='')
        img = cv.resize(frame, (0, 0), None, scale, scale, cv.INTER_AREA)
        for i in np.arange(0, img.shape[0] - window_size[0], stride):
            for j in np.arange(0, img.shape[1] - window_size[1], stride):
                patch = img[i:i + window_size[0], j:j + window_size[1], :]
                score = clf.decision_function([get_descriptor(patch)])[0]
                if score > threshold:
                    pt1 = np.array([i, j]) / scale
                    pt2 = np.array([i + window_size[0], j + window_size[1]]) / scale
                    if frame_scores[tuple(pt1.astype(int))] < score:
                        frame_scores[tuple(pt1.astype(int))] = score
                        frame_pos[tuple(pt1.astype(int))] = tuple(pt2.astype(int))
    peaks = feature.peak_local_max(frame_scores, 3 * stride)
    for peak in peaks:
        # print(frame_scores[tuple(peak)])
        cv.rectangle(frame_res, tuple(peak)[::-1], tuple(frame_pos[tuple(peak)][::-1]), (255, 0, 0), 5)
    return frame_res


img1 = cv.imread('./data/hw4/Melli.jpg')
img2 = cv.imread('./data/hw4/Persepolis.jpg')
img3 = cv.imread('./data/hw4/Esteghlal.jpg')

img1_det = FaceDetection(sgd, img1, scale_range=(1, 2), threshold=sgd_threshold)
img2_det = FaceDetection(sgd, img2, scale_range=(0.5, 1.5), threshold=sgd_threshold)
img3_det = FaceDetection(sgd, img3, scale_range=(0.5, 1.5), threshold=sgd_threshold)
# %%
cv.imwrite('./hw4/assignment/out/res4.jpg', img1_det)
cv.imwrite('./hw4/assignment/out/res5.jpg', img2_det)
cv.imwrite('./hw4/assignment/out/res6.jpg', img3_det)
# %%
plt.imshow(FaceDetection(sgd, img3, scale_range=(0.5, 2))[:, :, ::-1])
plt.show()
