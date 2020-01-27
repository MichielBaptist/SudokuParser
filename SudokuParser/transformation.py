import numpy as np
import cv2 as cv
import utils as ut
from functools import reduce
import imutils
import matplotlib.pyplot as plt

class Transformation:

    def __init__(self, debug = False, params = {}):
        self.debug = debug
        self.setName()
        self.setParams(params)

    def apply(self, original, img):
        pass

    def logStr(self, str):
        return f"{self.getName()} - {str}"

    def ifDebug(self, fn):

        if self.debug:
            fn()

    def getParamList(self):
        lst = self.getParamListNCV()
        lst = [(n, a, l, h, d, self.attrValue(a)) for (n,a,l,h,d) in lst]
        return lst

    def getParamListNCV(self):
        pass

    def setParams(self, params = {}):
        for param in self.getParamListNCV():
            self.setParam(param, params)

    def setParam(self, param, params = {}):
        n, a, l, h, d = param
        val = d if a not in params else params[a]
        setattr(self, a, val)

    def attrValue(self, attr):
        return None if not hasattr(self, attr) else getattr(self, attr)

class findDigitTrn(Transformation):
    def setName(self):
        self.name = "Digit"

    def getName(self):
        return self.name

    def threshMethod(self):
        if self.gaussian:
            m = cv.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            m = cv.ADAPTIVE_THRESH_MEAN_C
        return m

    def getParamListNCV(self):
        return [
            (self.logStr("Gaussian"), "gaussian", 0, 1, 1),
            (self.logStr("Block Size"), "block_size", 0, 51, 7),
            (self.logStr("C"), "c", 0, 50, 10),
            (self.logStr("Opening"), "opening", 0, 20, 3 )
        ]

    def apply(self, original, img):

        # 1) thresholding to find the digit
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray,
                                      255,
                                      self.threshMethod(),
                                      cv.THRESH_BINARY_INV,
                                      self.block_size,
                                      self.c)

        kernel = np.ones((self.opening, self.opening), np.uint8)
        eroded = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

        contours, hierarchy= cv.findContours(eroded,
                                   cv.RETR_TREE,
                                   cv.CHAIN_APPROX_SIMPLE)

        #self.ifDebug(lambda : self.paintContours(img, contours, sudoku_contour, corners))
        self.ifDebug(lambda : cv.imshow(self.logStr("Digit"), thresh))
        self.ifDebug(lambda : cv.imshow(self.logStr("Opening"), eroded))

        return

    def paintContours(self, img, contours, sudoku_contour, corners):
        img_cp = np.array(img)

        cv.drawContours(img_cp, contours, -1,ut.BLUE, thickness = 2)
        cv.drawContours(img_cp, [sudoku_contour], -1, ut.RED, thickness = 2)

        for cnr in corners:
            cv.circle(img_cp, tuple(cnr), 5, ut.GREEN, thickness = -1)

        cv.imshow(self.logStr("Contours"), img_cp)

class ChopTrn(Transformation):
    def setName(self):
        self.name = "Chop"

    def getName(self):
        return self.name

    def getParamListNCV(self):
        return [
            (self.logStr("Padding"), "padding", 0, 50, -2)
        ]

    def imgBlock_index(self, i, j, bs_h, bs_w, img):
        l = (int)(i*bs_w)
        r = (int)((i+1)*bs_w)
        u = (int)(j*bs_h)
        d = (int)((j+1)*bs_h)

        return self.imgBlock(u,d,l,r,img)

    def imgBlock(self, u, d, l, r, img):
        u, d, l, r = self.blockPadding(u,d,l,r, img)
        return img[u:d, l:r]

    def blockPadding(self, u, d, l, r, img):
        h = img.shape[0]
        w = img.shape[1]

        l = max(l - self.padding, 0)
        r = min(r + self.padding, w - 1)
        u = max(u - self.padding, 0)
        d = min(d + self.padding, h - 1)

        return u,d,l,r

    def apply(self, original, img):
        block_size = (img.shape[0]/9, img.shape[1]/9)
        blocks = [[(i,j) for i in range(9)] for j in range(9)]
        blocks = [[self.imgBlock_index(*b, *block_size, img) for b in row]for row in blocks]

        self.ifDebug(lambda : cv.imshow(self.logStr("block"),blocks[0][3]))
        return blocks

class RotateTrn(Transformation):
    def getParamListNCV(self):
        return [
            (self.logStr("Angle"), "rotation_angle", 0, 180, 0),
            (self.logStr("Cutaway/full"), "rotation_cut_away", 0, 1, 0)
        ]

    def setName(self):
        self.name = "Rotation"

    def getName(self):
        return self.name

    def apply(self, original, img):
        if self.rotation_cut_away:
            transformed = imutils.rotate(img, self.rotation_angle)
        else:
            transformed = imutils.rotate_bound(img, self.rotation_angle)

        return transformed

class GridLinesTrn(Transformation):
    def setName(self):
        self.name = "Grid lines of 9x9"

    def getName(self):
        return self.name

    def getParamListNCV(self):
        return [
            (self.logStr("Grid line width"), "grid_thick", 0, 10, 3)
        ]

    def drawHlines(self, img):
        h, w, _ = img.shape
        hf = h/9

        for i in range(1, 9):
            hi = (int)(hf*i)
            cv.line(img, (0,hi), (w, hi), ut.RED, 2)

        return np.array(img)

    def drawVlines(self, img):
        h, w, _ = img.shape
        wf = w/9

        for i in range(1, 9):
            wi = (int)(wf*i)
            cv.line(img, (wi,0), (wi, h), ut.RED, 2)

        return np.array(img)

    def apply(self, original, img):
        gridded = self.drawVlines(self.drawHlines(np.array(img)))

        self.ifDebug(lambda : cv.imshow(self.logStr("Grid"), gridded))

        return gridded

class IdentityTrn(Transformation):
    def apply(self, original, img):
        return img

class CropTrn(Transformation):

    def setName(self):
        self.name = "Crop"

    def getName(self):
        return self.name

    def getParamListNCV(self):
        return [
            (self.logStr("Edge detection: Canny/thresholding"), "canny_thresh", 0, 1, 1),
            (self.logStr("Canny: upper threshold"), "canny_max", 0, 1000, 650),
            (self.logStr("Canny: lower threshold"), "canny_min", 0, 1000, 250),
            (self.logStr("Thresholding: Mean/Gaussian"), "thresh_gaussian",  0, 1, 1),
            (self.logStr("Thresholding: Block size"), "thresh_block_size",0, 70, 21),
            (self.logStr("Thresholding: C"), "thresh_C",0, 100, 7),
            (self.logStr("Sobel window"), "sobel_window", 1, 15, 3),
            (self.logStr("Sobel dx"), "sobel_dx", 0, 9, 1),
            (self.logStr("Sobel dy"), "sobel_dy", 0, 9, 1)
        ]

    def threshMethod(self):
        if self.thresh_gaussian:
            m = cv.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            m = cv.ADAPTIVE_THRESH_MEAN_C
        return m

    def findEdges(self, img):
        if not self.canny_thresh:
            # Find edges with Canny()
            edges = cv.Canny(img,
                             self.canny_min,
                             self.canny_max)
        else:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            edges = cv.adaptiveThreshold(gray,
                                         255,
                                         self.threshMethod(),
                                         cv.THRESH_BINARY_INV,
                                         self.thresh_block_size,
                                         self.thresh_C)

        return edges

    def apply(self, original, img):
        # To crop the image to only the sudoku squarem:

        img_cp = np.array(img)

        # 1) Find edges in the sudoku
        edges = self.findEdges(img_cp)

        self.ifDebug(lambda : cv.imshow(self.logStr("Canny edges"), edges))

        # 2) find contours
        contours, hierarchy = cv.findContours(edges,
                                              cv.RETR_TREE,
                                              cv.CHAIN_APPROX_SIMPLE)

        # 3) Find the sudoku contour
        sudoku_contour = self.findSudokuContour(contours)

        # 4) Find the corners of the contour
        corners = self.findCorners(sudoku_contour)

        self.ifDebug(lambda : self.paintContours(img_cp, contours, sudoku_contour, corners))

        # 5) Warp the perspective of the sudoku
        perspective, width, height = self.findSudokuPerspective(corners)
        warped_img = cv.warpPerspective(img_cp, perspective, (width, height))

        self.ifDebug(lambda : cv.imshow(self.logStr("Warped image"), warped_img))

        return warped_img

    def paintContours(self, img, contours, sudoku_contour, corners):
        img_cp = np.array(img)

        cv.drawContours(img_cp, contours, -1,ut.BLUE, thickness = 2)
        cv.drawContours(img_cp, [sudoku_contour], -1, ut.RED, thickness = 2)

        for cnr in corners:
            cv.circle(img_cp, tuple(cnr), 5, ut.GREEN, thickness = -1)

        cv.imshow(self.logStr("Contours"), img_cp)

    def findCorners(self, cnt):
        bl = ut.min(cnt[:,0,:], lambda x: x[0] - x[1])
        tl = ut.min(cnt[:,0,:], lambda x: x[0] + x[1])
        br = ut.min(cnt[:,0,:], lambda x: -x[0]- x[1])
        tr = ut.min(cnt[:,0,:], lambda x: -x[0]+ x[1])

        return tl, tr, br, bl

    def findSudokuPerspective(self, frm):
        tl, tr, br, bl = frm

        new_width = (int)(max(ut.eucl(tl, tr), ut.eucl(bl, br)) - 1)
        new_height= (int)(max(ut.eucl(tr, br), ut.eucl(tl, bl)) - 1)

        to = [
            [0,                   0],
            [new_width,           0],
            [new_width,  new_height],
            [0,          new_height]
        ]
        perspective =  cv.getPerspectiveTransform(np.float32(frm), np.float32(to))
        return perspective, new_width, new_height

    def findSudokuContour(self, contours, edges = None, hierarchy = None):
        return ut.max(contours, cv.contourArea)

class ResizeTrn(Transformation):

    def setName(self):
        self.name = "Resize"

    def getName(self):
        return self.name

    def getParamListNCV(self):
        return [
            (self.logStr("Scale"), "scale", 0, 100, 100),
            (self.logStr("Maximum width"), "mx_ver",0, 2000, 1000),
            (self.logStr("Maximum height"), "mx_hor",0, 2000, 1000)
        ]

    def getScale(self):
        return self.scale / 100

    def apply(self, original, img):
        new_img = np.array(img)

        r, c, _ = new_img.shape

        hscale = self.calcScale(c, self.mx_hor)
        vscale = self.calcScale(r, self.mx_ver)

        scale = min(hscale, vscale, self.getScale())

        self.ifDebug(lambda : self.logStr(f"Scale: {scale}"))

        return self.scaleImg(new_img, scale)

    def scaleImg(self, img, scale):
        nr = (int)(img.shape[0] * scale)
        nc = (int)(img.shape[1] * scale)
        return cv.resize(img, (nr,nc))

    def calcScale(self, frm, to):
        return (to/frm)

def applyTransformations(img, trns):
    return reduce(lambda x, y: y.apply(np.array(img), x), trns, np.array(img))
