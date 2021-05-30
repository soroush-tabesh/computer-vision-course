import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import os


def find_train_test_files(root_path, train_dir='Train', test_dir='Test'):
    category_names = []
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    for root, dirs, files in os.walk(root_path):
        fs = root.replace(root_path, '').split(os.sep)
        if len(fs) != 2:
            continue
        category_name = fs[1]
        if category_name not in category_names:
            category_names.append(category_name)
        category_index = category_names.index(category_name)
        for f in files:
            img = cv.imread(os.path.join(root, f), cv.IMREAD_GRAYSCALE)
            if fs[0] == train_dir:
                train_imgs.append(img)
                train_labels.append(category_index)
            elif fs[0] == test_dir:
                test_imgs.append(img)
                test_labels.append(category_index)
    return train_imgs, np.array(train_labels), test_imgs, np.array(test_labels), category_names


root_path = './data/hw3/Data/'

train_imgs, train_labels, test_imgs, test_labels, category_names = find_train_test_files(root_path)


# %%
def simple_knn_predictor(train_imgs, train_labels, test_imgs, test_labels, knn_val, f_size,
                         weights='distance', metric='minkowski', p=1):
    train_imgs_d = np.array([cv.resize(img, (f_size, f_size), interpolation=cv.INTER_AREA).ravel()
                             for img in train_imgs])
    test_imgs_d = np.array([cv.resize(img, (f_size, f_size), interpolation=cv.INTER_AREA).ravel()
                            for img in test_imgs])

    accs = []
    skf = StratifiedKFold()
    for train_idx, valid_idx in skf.split(train_imgs_d, train_labels):
        clf = KNeighborsClassifier(n_neighbors=knn_val, weights=weights, metric=metric, p=p)
        clf.fit(train_imgs_d[train_idx], train_labels[train_idx])
        accs.append(accuracy_score(train_labels[valid_idx], clf.predict(train_imgs_d[valid_idx])))

    clf = KNeighborsClassifier(n_neighbors=knn_val, weights=weights, metric=metric, p=p)
    clf.fit(train_imgs_d, train_labels)
    pred = clf.predict(test_imgs_d)

    return pred, np.average(accs), accuracy_score(test_labels, pred)


f_size = 8
knn_val = 17
accs = []
for i in range(4, 20, 2):
    acc = simple_knn_predictor(train_imgs, train_labels, test_imgs, test_labels,
                               knn_val=1, f_size=i, weights='uniform')[1]
    accs.append((i, acc))
    print(i)
accs = np.array(accs)
plt.plot(accs[:, 0], accs[:, 1])
plt.show()

# best result
simple_knn_predictor(train_imgs, train_labels, test_imgs, test_labels, 17, 8, 'distance')

# %%
ft_cnt = 100
sift = cv.SIFT_create(ft_cnt)
train_imgs_words = np.vstack([sift.detectAndCompute(img, None)[1] for img in train_imgs])

# %%
clus_cnt = 80
train_imgs_dict = KMeans(clus_cnt)
train_imgs_dict.fit(train_imgs_words)


# %%
def extract_word_hist(imgs_list, word_dict):
    imgs_word_hist = []
    sift = cv.SIFT_create()
    for img in imgs_list:
        img_ft = sift.detectAndCompute(img, None)[1]
        img_words = word_dict.predict(img_ft).tolist()
        hist = np.array([img_words.count(i) for i in range(word_dict.n_clusters)])
        hist = hist / hist.sum()
        imgs_word_hist.append(hist)
    return np.array(imgs_word_hist)


train_imgs_hist = extract_word_hist(train_imgs, train_imgs_dict)
test_imgs_hist = extract_word_hist(test_imgs, train_imgs_dict)

# %%
accs = []
skf = StratifiedKFold()
for train_idx, valid_idx in skf.split(train_imgs_hist, train_labels):
    clf = KNeighborsClassifier(n_neighbors=11, p=1)
    clf.fit(train_imgs_hist[train_idx], train_labels[train_idx])
    accs.append(accuracy_score(train_labels[valid_idx], clf.predict(train_imgs_hist[valid_idx])))
print('train accuracy:', np.average(accs))

clf = KNeighborsClassifier(n_neighbors=11, p=1)
clf.fit(train_imgs_hist, train_labels)
print('test accuracy:', accuracy_score(test_labels, clf.predict(test_imgs_hist)))

# %%
accs = []
skf = StratifiedKFold()
for train_idx, valid_idx in skf.split(train_imgs_hist, train_labels):
    clf = SVC(C=1.2, kernel='rbf')
    clf.fit(train_imgs_hist[train_idx], train_labels[train_idx])
    accs.append(accuracy_score(train_labels[valid_idx], clf.predict(train_imgs_hist[valid_idx])))
print('train accuracy:', np.average(accs))

clf = SVC(C=1.2, kernel='rbf')
clf.fit(train_imgs_hist, train_labels)
print('test accuracy:', accuracy_score(test_labels, clf.predict(test_imgs_hist)))

plt.matshow(confusion_matrix(test_labels, clf.predict(test_imgs_hist)))
plt.show()
