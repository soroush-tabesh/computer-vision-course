import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
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
print('train accuracy:', 100 * np.average(accs))

clf = KNeighborsClassifier(n_neighbors=11, p=1)
clf.fit(train_imgs_hist, train_labels)
print('test accuracy:', 100 * accuracy_score(test_labels, clf.predict(test_imgs_hist)))
