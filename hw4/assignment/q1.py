#####
# positive image data is LFW
# negative image data is a cleaned-up version of CALTECH-256 which can be found bellow
# https://mega.nz/folder/c6YD2CxQ#upVP9Ybn2afcbp3r6R7WWA
#####

import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import feature
from sklearn import utils, linear_model, metrics
from sklearn.model_selection import train_test_split

positive_image_path = './data/hw4/pos'
negative_image_path = './data/hw4/neg'
output_path = './out'


def get_all_files(path, pat='jpg'):
    list_of_files = []
    for (dir_path, dir_names, filenames) in os.walk(path):
        list_of_files += [os.path.join(dir_path, file) for file in filenames
                          if file.split('.')[-1].lower() == pat.lower()]
    return list_of_files


def get_image(filename, size=(128, 128), margin=(50, 50, 50, 50), min_size=(150, 150)):
    src = cv.imread(filename)
    if src.shape[0] > min_size[0] and src.shape[1] > min_size[1]:
        src = src[margin[0]:-margin[1], margin[2]:-margin[3], :]
    return cv.resize(src, size, interpolation=cv.INTER_AREA)


def get_descriptor(src, cell_size=(8, 8), block_size=(2, 2), n_bins=9):
    hog = cv.HOGDescriptor(_winSize=(src.shape[1] // cell_size[1] * cell_size[1],
                                     src.shape[0] // cell_size[0] * cell_size[0]),
                           _blockSize=(block_size[1] * cell_size[1],
                                       block_size[0] * cell_size[0]),
                           _blockStride=(cell_size[1], cell_size[0]),
                           _cellSize=(cell_size[1], cell_size[0]),
                           _nbins=n_bins)
    return hog.compute(src).ravel()


# load data
X_file = []
tmp = utils.shuffle(get_all_files(positive_image_path), random_state=0)
X_file += tmp
X_file += utils.shuffle(get_all_files(negative_image_path), random_state=0)
y = np.zeros(len(X_file), dtype=np.int32)
y[:len(tmp)] = 1

# split data
(X_train_file, X_test_file,
 y_train, y_test) = train_test_split(X_file, y,
                                     stratify=y, test_size=2000, random_state=0, shuffle=True)
(X_train_file, X_valid_file,
 y_train, y_valid) = train_test_split(X_train_file, y_train,
                                      stratify=y_train, test_size=2000, random_state=0, shuffle=True)

# %% fit a gd linear svm
print('Fitting the Classifier')
sgd = linear_model.SGDClassifier(loss='hinge',
                                 penalty='l2',
                                 fit_intercept=True,
                                 alpha=0.0001,
                                 shuffle=False,
                                 learning_rate='optimal')
for i, (filename, label) in enumerate(zip(X_train_file, y_train)):
    ft = get_descriptor(get_image(filename))
    sgd.partial_fit([ft], [label], classes=[0, 1])


# %% evaluate model
def evaluate_model(clf, X_file, y):
    y_pred = np.zeros_like(y)
    y_score = np.zeros_like(y, dtype=float)
    for i, filename in enumerate(X_file):
        ft = get_descriptor(get_image(filename))
        y_pred[i] = clf.predict([ft])
        y_score[i] = clf.decision_function([ft])
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


# %% evaluate validation data and tune parameters
eval_valid = evaluate_model(sgd, X_valid_file, y_valid)
print(f'validation accuracy = {calculate_accuracy(y_valid, *eval_valid)}')
sgd_threshold = eval_valid[1][eval_valid[1] > 0].mean() - 2 * eval_valid[1][eval_valid[1] > 0].std()

# %% evaluate test data
eval_test = evaluate_model(sgd, X_test_file, y_test)
score_test = calculate_accuracy(y_test, *eval_test)
roc_test = calculate_roc(y_test, *eval_test)
roc_auc_test = calculate_roc_auc(y_test, *eval_test)
prc_test = calculate_prc(y_test, *eval_test)
ap_test = calculate_ap(y_test, *eval_test)

plt.plot(roc_test[0], roc_test[1])
plt.title(f'ROC, AUC={roc_auc_test:.5f}')
plt.savefig(os.path.join(output_path, 'res1.jpg'))
plt.clf()

plt.plot(prc_test[0], prc_test[1])
plt.title(f'Precision-Recall Curve, AP={ap_test:.5f}')
plt.savefig(os.path.join(output_path, 'res2.jpg'))
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
        cv.rectangle(frame_res, tuple(peak)[::-1], tuple(frame_pos[tuple(peak)][::-1]), (255, 0, 0), 5)
    return frame_res


print('Starting face detection')
img1 = cv.imread('./data/hw4/Melli.jpg')
img2 = cv.imread('./data/hw4/Persepolis.jpg')
img3 = cv.imread('./data/hw4/Esteghlal.jpg')

img1_det = FaceDetection(sgd, img1, scale_range=(1, 2), threshold=sgd_threshold)
img2_det = FaceDetection(sgd, img2, scale_range=(0.5, 1.5), threshold=sgd_threshold)
img3_det = FaceDetection(sgd, img3, scale_range=(0.5, 1.5), threshold=sgd_threshold)

cv.imwrite(os.path.join(output_path, 'res4.jpg'), img1_det)
cv.imwrite(os.path.join(output_path, 'res5.jpg'), img2_det)
cv.imwrite(os.path.join(output_path, 'res6.jpg'), img3_det)
