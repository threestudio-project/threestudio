import numpy as np
import cv2
import argparse
from PIL import Image
import os
import imageio
from tqdm import tqdm


def make_path_composite():
    path = "outputs/att3d-if/composite_prompts@20230804-194850/save"
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
        "outputs/att3d-if/composite_prompts@20230804-172543/save/a hamburger/it20000-test/0.png",
        "outputs/att3d-if/composite_prompts@20230804-172543/save/a pineapple/it20000-test/0.png"
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
    parser.add_argument("--save_dir", default="./image_grid.mp4", type=str)
    args = parser.parse_args()
    return args


def make_grid(paths, h, resize):
    imgs = []
    for path in paths:
        img = cv2.imread(path)
        img = img[:, :img.shape[1] // 3]
        img = np.array(Image.fromarray(img).resize((resize, resize)))
        imgs.append(img)
    
    h = int(max(h, 1))
    w = (len(imgs) + h - 1) // h
    grids = []
    for i in range(h):
        grid_row = imgs[(i*w):((i+1)*w)]
        if len(grid_row) < w:
            grid_row += [np.ones_like(imgs[0]) * 255 for _ in range(w - len(grid_row))]
        grids.append(np.concatenate(grid_row, axis=1))
    grids = np.concatenate(grids, axis=0)
    return grids


def make_video():
    args = setup()
    imgs = []
    paths = [os.path.dirname(path) for path in make_path_all()]
    num = np.inf
    for path in paths:
        num = min(num, len(os.listdir(path)))
    
    for i in tqdm(range(num)):
        path_frame = [os.path.join(path, f"{i}.png") for path in paths]
        imgs.append(make_grid(path_frame, args.h, args.resize, args.clip))

    imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
    imageio.mimsave(args.save_dir, imgs, fps=30)


make_video()
