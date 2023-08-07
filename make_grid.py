import numpy as np
import cv2
import argparse
from PIL import Image
import os


def make_path_composite():
    path = "outputs/att3d-if/composite_prompts@20230805-094945/save"
    themes = [
        "blue",
        "red",
        "green",
        "purple"
    ]

    subjects = [
        "a squirrel",
        "an airplane",
        "a hamburger",
        "a pineapple"
    ]

    paths = []
    for subject in subjects:
        for theme in themes:
            subpath = os.path.join(path, f"{subject}, {theme}")
            folder = sorted(os.listdir(subpath))[0]
            paths.append(os.path.join(subpath, folder, "0.png"))
    return paths


def make_path_list():
    paths = [
        "outputs/dreamfusion-if/a_pineapple,_blue@20230807-140650/save/it5000-test/0.png",
        "outputs/dreamfusion-if/a_hamburger,_green@20230807-140549/save/it5000-test/0.png"
    ]
    return paths


def make_path_all():
    path = "outputs/att3d-if/composite_prompts@20230805-165224/save"
    paths = []
    for folder in os.listdir(path):
        if folder == "val" or folder.endswith(".mp4"):
            continue
        
        subpath = sorted(os.listdir(os.path.join(path, folder)))[0]
        paths.append(os.path.join(path, folder, subpath, "0.png"))
    return paths


def setup():
    parser = argparse.ArgumentParser("Grid")
    parser.add_argument("--h", default=0, type=int)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--save_dir", default="./image_grid.png", type=str)
    args = parser.parse_args()
    return args


def make_grid():
    args = setup()
    imgs = []
    for path in make_path_all():
        img = cv2.imread(path)
        img = img[:, :img.shape[1] // 3]
        img = np.array(Image.fromarray(img).resize((args.resize, args.resize)))
        imgs.append(img)
    
    h = int(max(args.h, 1))
    w = (len(imgs) + h - 1) // h
    grids = []
    for i in range(h):
        grid_row = imgs[(i*w):((i+1)*w)]
        if len(grid_row) < w:
            grid_row += [np.ones_like(imgs[0]) * 255 for _ in range(w - len(grid_row))]
        grids.append(np.concatenate(grid_row, axis=1))
    grids = np.concatenate(grids, axis=0)
    cv2.imwrite(args.save_dir, grids)


make_grid()
