import sklearn
import numpy as np
import os
import cv2 as cv
import utils as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

class DigitRecognizer:

    def __init__(self):
        self.features = [
            np.mean,
            np.median,
            lambda x: self.meanThresh(x)
        ]
        self.rf = RandomForestClassifier()
        pass

    def train(self, X, Y):
        X = self.extractFeatures(X)

        self.rf.fit(X, Y)

        for i in range(10):
            nidxs = ut.getIndex(Y, lambda x: x==i)
            dat = np.take(X, nidxs, axis = 0)
            plt.plot(dat[:,2], dat[:,1], 'o', label = f"{i}")

        plt.legend()
        plt.show()

    def extractFeatures(self, X):
        return np.array(applyFsToXs(self.features, X))

    def predict(self, X):
        X = self.extractFeatures(X)
        return self.rf.predict(X)

    def meanThresh(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 3)
        return np.mean(thresh)

def applyFsToXs(Fs, Xs):
    return list(map(lambda x: applyFsToX(Fs, x), Xs))

def applyFsToX(Fs, X):
    return list(map(lambda f: f(X), Fs))

def loadFile(path):
    return cv.imread(path)

def loadDigitData(str, i):
    dir = os.path.join(str, f"{i}")
    files = [os.path.join(dir, f) for f in os.listdir(dir)]

    Xs = [loadFile(f) for f in files]
    Ys = [i for j in range(len(Xs))]


    return (Xs, Ys)

def loadData(str):
    sets = [loadDigitData(str, i) for i in range(10)]

    Xs = ut.flatten([X for X,Y in sets], 1)
    Ys = ut.flatten([Y for X,Y in sets], 1)

    return Xs, Ys

Xs, Ys = loadData("digits")
Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(Xs, Ys, test_size = 0.33)

rec = DigitRecognizer()

rec.train(Xs_train, Ys_train)

Ys_pred = rec.predict(Xs_test)
Ytest_nz, Ypred_nz = ut.mutFilter(Ys_test, lambda x: x != 0, Ys_pred)

print(accuracy_score(Ytest_nz, Ypred_nz))
print(accuracy_score(Ys_test, Ys_pred))
