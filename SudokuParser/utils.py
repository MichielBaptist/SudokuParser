import numpy as np
import functools
import shutil
import os

import cv2 as cv

RED = (0,0,255)
BLUE = (0,255,0)
GREEN = (255,0,0)
IMT = (1,0,2)
GS_MAX = 255

GAUSSIAN = cv.ADAPTIVE_THRESH_GAUSSIAN_C
MEAN = cv.ADAPTIVE_THRESH_MEAN_C

# Transposes an image (a cv2 image)
def imT(img):
    return np.transpose(np.array(img), IMT)


def flt2d(lst):
    """
    Flattens a list of lists:
        [[], ..., []] -> []

    This method will remove exactly the layer at index 1. This method
    assumes that every element in the list is a list itself.
    """
    return [e for sl in lst for e in sl]

def argmax(lst, fn = lambda x: x, default = None):
    if len(lst) == 0:
        return default

    return np.argmax([fn(e) for e in lst])

def max(lst, fn = lambda x: x, default = None):
    if len(lst) == 0:
        return default

    return lst[argmax(lst, fn)]

def min(lst, fn = lambda x: x, default = None):
    if len(lst) == 0:
        return default

    return lst[argmin(lst, fn)]

def argmin(lst, fn = lambda x: x, default = None):
    if len(lst) == 0:
        return default
        
    return np.argmin([fn(e) for e in lst])

def eucl(p1, p2):
    return np.linalg.norm(p1-p2)

def flatten(l, d = None):

    def pack(e):
        return [e] if not isinstance(e, list) else e

    if d == 0:
        return l
    elif d == None:
        nd = None
    else:
        nd = d - 1

    if isinstance(l, list):
        l = flt2d(map(pack, map(lambda x: flatten(x, nd), l)))

    return l

def take(lst, ixs):
    return [lst[i] for i in ixs]

# lst: list to be filtered
# fn: funtion to filter
# lsts: other mutual lists
def mutFilter(lst1, fn, lst2):
    ixs = getIndex(lst1, fn)
    return take(lst1, ixs), take(lst2, ixs)

def getIndex(lst, fn):
    return [i for i, e in enumerate(lst) if fn(e)]

def setIfIndex(lst, ids, val):
    # TODO: make this more efficient
    return [(val if i in ids else e) for i, e in enumerate(lst)]

def isDigit(str):
    try:
        toDigit(str)
        return True
    except:
        return False

def toDigit(str):
    d = int(str)
    if d not in range(10):
        raise Exception("Not digit")
    return d

def emptyStrct(strct):
    return strct == []

def getRoot(strct):
    return strct[0]

def getSubStrct(strct):
    return strct[1]

def deleteOutDirStructure(strct):
    root = getRoot(strct)

    if os.path.exists(root):
        shutil.rmtree(root)

def createOutDirStructure(strct, cd = ""):
    if emptyStrct(strct):
        return

    root = getRoot(strct)
    subStrct = getSubStrct(strct)

    root = os.path.join(cd, root)
    if not os.path.exists(root):
        os.mkdir(root)

    for subF  in subStrct:
        createOutDirStructure(subF, root)

def h(x):
    print("Hallo?")

"""Find the largest contour of the given image. If no contour is
found on the image, the contour is the edges of the image.

This method exclusively uses adaptiveThresholding."""
def findLargestContourAT(img,
                         method = GAUSSIAN,
                         window_size_fraction = 0.15,
                         window_size = 5,
                         threshold_C = 15):

    img_cp = np.array(img)
    if window_size == None:
        w, h, _ = img_cp.shape
        window_size = int( window_size_fraction *  min(w,h))

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.adaptiveThreshold(gray,
                                 GS_MAX,
                                 method,
                                 cv.THRESH_BINARY_INV,
                                 window_size,
                                 threshold_C)

    contours, hierarchy = cv.findContours(edges,
                                          cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)

    print(f"contours: {len(contours)}")
    img_cp = cv.drawContours(img_cp, contours, -1, GREEN)
    cv.imshow("img", img_cp)
    cv.waitKey()

    if contours == []:
        print(None)

    largest_contour = ut.max(contours, cv.contourArea)

    return largest_contour
