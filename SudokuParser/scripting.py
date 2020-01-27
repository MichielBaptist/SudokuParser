import parse
import cv2 as cv
import loading
import os

import utils as ut

import matplotlib.pyplot as plt

def parseLabels(f):
    return [[e.strip() for e in r.split(" ")] for r in f]

# Where is the data?
s_x = "sudokus/x"
s_y = "sudokus/y"

# Where should the chopped digits go?
out_n = "digits"
out_f = (out_n, [(f"{i}",[]) for i in range(10)])

ut.deleteOutDirStructure(out_f)
ut.createOutDirStructure(out_f)

# Read all sudokus and labels
suds_x = sorted(os.listdir(s_x))
suds_y = sorted(os.listdir(s_y))

print(f"Putting digits in {out_n}")

for (x_p, y_p) in zip(suds_x, suds_y):
    # Chop sudoku and attatch label
    xs = parse.chopSudoku_path(os.path.join(s_x, x_p))
    ys = parseLabels(open(os.path.join(s_y, y_p)))

    print(f"Chopping sudoku: {x_p}")

    for xr, yr in zip(xs,ys):
        for x,y in zip(xr,yr):
            x_dir = os.path.join(out_n, str(y))
            num = len(os.listdir(x_dir))
            x_file = os.path.join(x_dir, f"{num}.png")

            cv.imwrite(x_file, img=x)

print(f"Done parsing {len(list(zip(suds_x, suds_y)))} sudokus!")
for i in range(10):
    pth = os.path.join(out_n, str(i))
    print(f"Number of {i} images: {len(os.listdir(pth))}")
