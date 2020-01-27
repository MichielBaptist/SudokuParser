import cv2 as cv
from functools import reduce

import utils as ut
import transformation as trns
import loading

import numpy as np

def parseSudoku_path(path, params = {}, debug = False):
    return parseSudoku(loading.loadImage(path), params)

def chopSudoku_path(path, params = {}, debug = False):
    return chopSudoku(loading.loadImage(path), params)

def chopSudoku(img, params = None, debug = False):
    params["debug"] = debug

    # 1) Find the sudoku in the image.
    crop_transformations = getCropTrns(params)
    cropped_sudoku = trns.applyTransformations(img, crop_transformations)

    # 2) Chop up the cropped image
    chop_transformations = getChopTrns(params)
    chopped_sudoku = trns.applyTransformations(cropped_sudoku, chop_transformations )

    return chopped_sudoku

def parseSudoku(img, params = None, debug = False):
    params["debug"] = debug

    # 1) Find the sudoku in the image.
    crop_transformations = getCropTrns(params)
    cropped_sudoku = trns.applyTransformations(img, crop_transformations)

    # 2) Chop up the cropped image
    chop_transformations = getChopTrns(params)
    chopped_sudoku = trns.applyTransformations(cropped_sudoku, chop_transformations )

    #piece = chopped_sudoku[0][3]
    #digit_trns = getDigitTrns(params)
    #trns.applyTransformations(piece, digit_trns)

    # 3) Digit recognition
    #parsed_sudoku = parseChoppedSudoku(chopped_sudoku)
    # Return parsed sudoku
    return

def parseChoppedSudoku(chopped):
    return [[parseSudokuPiece(p) for p in row] for row in chopped]

def parseSudokuPiece(piece_img):
    # Parse the digits here
    return 0

def getAllParameters():
    lst = ut.flatten([t.getParamList() for t in getCropTrns()])
    lst+= ut.flatten([t.getParamList() for t in getChopTrns()])
    #lst+= ut.flatten([t.getParamList() for t in getDigitTrns()])
    return lst

def getDigitTrns(params = {}):
    return [
        trns.findDigitTrn(params = params)
    ]

def getChopTrns(params = {}):
    return [
        trns.ChopTrn(params = params)
    ]

def getCropTrns(params = {}):
    return [
        #trns.RotateTrn(params = params),        # Debug and test
        trns.ResizeTrn(params = params),        # Resize
        trns.CropTrn(params = params),          # Crop to sudoku
        #trns.GridLinesTrn(params = params)      # Debug and test
    ]

def reparse(sudoku, params):
    parseSudoku(sudoku, params)

def paramChange(name, params, sudoku):
    return lambda val: setParamValueAndReparse(name, val, params, sudoku)

def setParamValueAndReparse(name, val, params, sudoku):
    print(f"Param: {name} changed to {val}")
    for k,v in params.items():
        print(f"{k} --> {v}")
    params[name] = val
    reparse(sudoku, params)


"""
pth = "sudokus/x/12.jpg"
sud = loading.loadImage(pth)

cv.namedWindow("Parameters")
params = getAllParameters()
params_dict = {a:c for _,a,_,_,_,c in params}
for n, a, l, h, d, c in params:
    cv.createTrackbar(n, "Parameters", c, h, paramChange(a, params_dict, sud))

cv.imshow("Parameters", np.zeros((1,400)))
reparse(sud, params_dict)

cv.waitKey()
"""
