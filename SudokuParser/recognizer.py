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

    def fit(self, X, Y):

        # 1) Preprocess the data:
        X = self.resizeAll(X)
        X = np.array(X)
        X = X[:,4:28,8:24,:]

        # 2) Separate digits from non digits
        # TODO: add utility to take using numpy not list comprehensions?
        Ids = ut.getIndex(Y, lambda x: x != 0)
        Y_binary = ut.setIfIndex(Y, Ids, 1)

        # 3) train a binary classifier
        # 3.1) convert to grayscale
        X_gray = applyFToXs(
            lambda img: cv.cvtColor(img, cv.COLOR_BGR2GRAY),
            X)

        X_threshed = applyFToXs(
            lambda img: cv.adaptiveThreshold(img,
                                             255,
                                             cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY_INV,
                                             blockSize = 3,
                                             C = 5),
            X_gray
        )

        contours = applyFToXs(
            lambda x: cv.findContours(
                x,
                cv.RETR_TREE,
                cv.CHAIN_APPROX_SIMPLE
            )[0],
            X_threshed
        )
        max_contours = applyFToXs(findMaxContour, contours)
        bounding_rects = applyFToXs(findBindingRect, max_contours)
        areas = applyFToXs(findRectArea, bounding_rects)


        cv.imshow("1", X_threshed[0])
        cv.imshow("2", X_threshed[1])
        cv.waitKey()

        print(np.array(X_threshed).shape)
        means = applyFToXs(np.mean, X_threshed)
        #print(means)
        plt.plot(areas, Y_binary, 'd')
        plt.show()
        quit()


        X_flat = np.reshape(X, ())

        X = self.extractFeatures(X)
        self.rf.fit(X, Y)


    def showDebug(self):
        for i in range(10):
            nidxs = ut.getIndex(Y, lambda x: x==i)
            dat = np.take(X, nidxs, axis = 0)
            plt.plot(dat[:,2], dat[:,1], 'o', label = f"{i}")

        plt.legend()
        plt.show()


    def resizeAll(self, X):
        new_size = (32, 32)
        F = lambda x: cv.resize(x,
                                dsize = new_size,
                                interpolation = cv.INTER_LINEAR)
        return applyFToXs(F, X)

    def preProcess(self, X):
        # 1)

        X = applyFToXs(lambda x: cv.resize(x, dsize = new_size, interpolation = cv.INTER_LINEAR), X)
        #print(X)
        # 1) crop to the digit
        print(len(X))
        print(np.array(X).shape)
        quit()
        digitContours = list(map(ut.findLargestContourAT, X))
        #cropRect = cv.boundingRect(digitContours)
        print("Here")
        # 2) Resize digit to common size
        return X

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

def applyFToXs(F, Xs):
    return list(map(F, Xs))

def loadFile(path):
    return cv.imread(path)

def loadDigitData(str, i):
    dir = os.path.join(str, f"{i}")
    files = [os.path.join(dir, f) for f in os.listdir(dir)]

    Xs = [loadFile(f) for f in files]
    Ys = [i for j in range(len(Xs))]


    return (Xs, Ys)

def findMaxContour(cntrs):
    if len(cntrs) == 0:
        return []

    return ut.max(cntrs, cv.contourArea)

def findBindingRect(cntr):
    if len(cntr) == 0:
        return (0,0,0,0)

    return cv.boundingRect(cntr)

def findRectArea(rect):
    _, _, w, h = rect
    return w*h

def loadData(str):
    sets = [loadDigitData(str, i) for i in range(10)]

    Xs = ut.flatten([X for X,Y in sets], 1)
    Ys = ut.flatten([Y for X,Y in sets], 1)

    return Xs, Ys

Xs, Ys = loadData("digits")
Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(Xs, Ys, test_size = 0.33)

rec = DigitRecognizer()

rec.fit(Xs_train, Ys_train)

Ys_pred = rec.predict(Xs_test)
Ytest_nz, Ypred_nz = ut.mutFilter(Ys_test, lambda x: x != 0, Ys_pred)

print(accuracy_score(Ytest_nz, Ypred_nz))
print(accuracy_score(Ys_test, Ys_pred))
