# make_training_vid("outputs/zero123/64_teddy_rgba.png@20230627-195615", frames_per_vid=30, fps=20, max_iters=200)
import argparse
import glob
import os

import imageio
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def draw_text_in_image(img, texts):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    black, white = (0, 0, 0), (255, 255, 255)
    for i, text in enumerate(texts):
        draw.text((2, (img.size[1] // len(texts)) * i + 1), f"{text}", white)
        draw.text((0, (img.size[1] // len(texts)) * i + 1), f"{text}", white)
        draw.text((2, (img.size[1] // len(texts)) * i - 1), f"{text}", white)
        draw.text((0, (img.size[1] // len(texts)) * i - 1), f"{text}", white)
        draw.text((1, (img.size[1] // len(texts)) * i), f"{text}", black)
    return np.asarray(img)


def make_training_vid(exp, frames_per_vid=1, fps=3, max_iters=None, max_vids=None):
    # exp = "/admin/home-vikram/git/threestudio/outputs/zero123/64_teddy_rgba.png@20230627-195615"
    files = glob.glob(os.path.join(exp, "save", "*.mp4"))
    if os.path.join(exp, "save", "training_vid.mp4") in files:
        files.remove(os.path.join(exp, "save", "training_vid.mp4"))
    its = [int(os.path.basename(file).split("-")[0].split("it")[-1]) for file in files]
    it_sort = np.argsort(its)
    files = list(np.array(files)[it_sort])
    its = list(np.array(its)[it_sort])
    max_vids = max_iters // its[0] if max_iters is not None else max_vids
    files, its = files[:max_vids], its[:max_vids]
    frames, i = [], 0
    for it, file in tqdm(zip(its, files), total=len(files)):
        vid = imageio.mimread(file)
        for _ in range(frames_per_vid):
            frame = vid[i % len(vid)]
            frame = draw_text_in_image(frame, [str(it)])
            frames.append(frame)
            i += 1
    # Save
    imageio.mimwrite(os.path.join(exp, "save", "training_vid.mp4"), frames, fps=fps)


def join(file1, file2, name):
    # file1 = "/admin/home-vikram/git/threestudio/outputs/zero123/OLD_64_dragon2_rgba.png@20230629-023028/save/it200-val.mp4"
    # file2 = "/admin/home-vikram/git/threestudio/outputs/zero123/64_dragon2_rgba.png@20230628-152734/save/it200-val.mp4"
    vid1 = imageio.mimread(file1)
    vid2 = imageio.mimread(file2)
    frames = []
    for f1, f2 in zip(vid1, vid2):
        frames.append(
            np.concatenate([f1[:, : f1.shape[0]], f2[:, : f2.shape[0]]], axis=1)
        )
    imageio.mimwrite(name, frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="directory of experiment")
    parser.add_argument(
        "--frames_per_vid", type=int, default=1, help="# of frames from each val vid"
    )
    parser.add_argument("--fps", type=int, help="max # of iters to save")
    parser.add_argument("--max_iters", type=int, help="max # of iters to save")
    parser.add_argument(
        "--max_vids",
        type=int,
        help="max # of val videos to save. Will be overridden by max_iters",
    )
    args = parser.parse_args()
    make_training_vid(
        args.exp, args.frames_per_vid, args.fps, args.max_iters, args.max_vids
    )
