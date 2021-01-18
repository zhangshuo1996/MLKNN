"""

"""
import numpy as np
import os
from tqdm import tqdm
import time

from skmultilearn.adapt import MLkNN2
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score


def HammingLoss(test_y, predict):
    label_num = test_y.shape[1]
    test_data_num = test_y.shape[0]
    hamming_loss = 0
    temp = 0
    predict = np.array(predict)
    for i in range(test_data_num):
        temp = temp + np.sum(test_y[i] ^ predict[i])

    hamming_loss = temp / label_num / test_data_num

    return hamming_loss


def convert_prediction(old_predict):
    """

    :param old_predict:
    :return:
    """
    label_num = old_predict.shape[1]
    res = []
    for sample_pre in old_predict.rows:
        tmp = np.zeros(label_num, dtype=int)
        for index in sample_pre:
            tmp[index] = 1
        res.append(tmp)
    return np.array(res)


def use_sklearn_ml_knn():
    """

    :return:
    """


    base_path = os.getcwd()
    # train_x = np.load(os.path.join(base_path, 'dataset/train_x.npy'), allow_pickle=True)
    # train_y = np.load(os.path.join(base_path, 'dataset/train_y.npy'), allow_pickle=True)

    train_x = np.load(os.path.join(base_path, 'my_dataset/train_x.npy'), allow_pickle=True)
    train_y = np.load(os.path.join(base_path, 'my_dataset/train_y.npy'), allow_pickle=True)

    new_train_y = []
    for tup in train_y:
        tmp = []
        for label in tup:
            if label == 0:
                tmp.append(0)
            else:
                tmp.append(1)
        new_train_y.append(tmp)

    # test_x = np.load('dataset/test_x.npy', allow_pickle=True)
    # test_y = np.load('dataset/test_y.npy', allow_pickle=True)

    test_x = np.load('my_dataset/test_x.npy', allow_pickle=True)
    test_y = np.load('my_dataset/test_y.npy', allow_pickle=True)
    new_test_y = []
    for tup in test_y:
        tmp = []
        for label in tup:
            if label == 0:
                tmp.append(0)
            else:
                tmp.append(1)
        new_test_y.append(tmp)

    new_test_y = np.array(new_test_y)

    classifier = MLkNN2(train_x, np.array(new_train_y), k=10)

    # classifier.fit(train_x, np.array(new_train_y))
    classifier.fit()
    predictions = classifier.predict(test_x)
    predictions = convert_prediction(predictions)

    # hamming_loss = HammingLoss(new_test_y, predictions)
    h_loss = hamming_loss(new_test_y, predictions)
    z = zero_one_loss(new_test_y, predictions)
    c = coverage_error(new_test_y, predictions)
    r = label_ranking_loss(new_test_y, predictions)
    a = average_precision_score(new_test_y, predictions)
    print('hamming_loss = ', h_loss)
    print('0-1_loss = ', z)
    print('cover_loss = ', c)
    print('rank_loss = ', r)
    print('average_loss = ', a)


if __name__ == '__main__':
    start = time.time()
    use_sklearn_ml_knn()
    end = time.time()
    print("spend time ", (end-start))