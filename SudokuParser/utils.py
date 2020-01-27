import numpy as np
import functools
import shutil
import os

RED = (0,0,255)
BLUE = (0,255,0)
GREEN = (255,0,0)
IMT = (1,0,2)

# Transposes an image (a cv2 image)
def imT(img):
    return np.transpose(np.array(img), IMT)


def flt2d(lst):
    """Flattens a list of lists: [[], ..., []] -> []"""
    return [e for sl in lst for e in sl]

def argmax(lst, fn = lambda x: x):
    return np.argmax([fn(e) for e in lst])

def max(lst, fn = lambda x: x):
    return lst[argmax(lst, fn)]

def min(lst, fn = lambda x: x):
    return lst[argmin(lst, fn)]

def argmin(lst, fn = lambda x: x):
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
