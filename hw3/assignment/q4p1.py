import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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


# best result using nn
nn_best = simple_knn_predictor(train_imgs, train_labels, test_imgs, test_labels,
                               knn_val=1, f_size=8, p=1)
print('Nearest Neighbor:')
print(f'    train accuracy is {100 * nn_best[1]:.2f} and test accuracy is {100 * nn_best[2]:.2f}')

# best result using knn
knn_value = 6
knn_best = simple_knn_predictor(train_imgs, train_labels, test_imgs, test_labels,
                                knn_val=knn_value, f_size=8, metric='distance', p=1)
print(f'K Nearest Neighbor: with k={knn_value}')
print(f'    train accuracy is {100 * knn_best[1]:.2f} and test accuracy is {100 * knn_best[2]:.2f}')
