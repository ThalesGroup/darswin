#!/usr/bin/env python3

import sys
import math
import pathlib
import cv2

sys.path.append('..')

from utils import distort_image

IN_PATH = "./tiny-imagenet-200"
OUT_PATH = "./tiny-imagenet-200-fisheye"

ALPHA = math.pi/4.5
D = [0.5, 0.5, 0.5, 0.5]


def main():
    convert_dataset(IN_PATH, OUT_PATH, ALPHA, D)
    # convert_image(IN_PATH, OUT_PATH, ALPHA, D)

def convert_image(fpath, out_dir, alpha, params):
    fpath = pathlib.Path(fpath)
    out_dir = pathlib.Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir.joinpath(fpath.name)

    print('Creating', outpath)

    img = cv2.imread(str(fpath))
    distorted = distort_image(img, alpha, params)
    cv2.imwrite(str(outpath), distorted)


def convert_dataset(in_dir, out_dir, alpha, params):
    in_dir = pathlib.Path(in_dir)
    out_dir = pathlib.Path(out_dir)

    for fpath in in_dir.rglob('*'):
        if fpath.suffix == '.JPEG':
            relpath = fpath.parent.relative_to(in_dir)
            outpath = out_dir.joinpath(relpath)
            convert_image(fpath, outpath, alpha, params)


if __name__=='__main__':
    main()