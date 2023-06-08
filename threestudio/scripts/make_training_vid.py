import argparse
import glob
import imageio
import os

from PIL import Image, ImageDraw
from tqdm import tqdm


def run(path, num):
    vids = glob.glob(os.path.join(path, "save/*.mp4"))
    idx = [
        int(os.path.basename(vid).split(".mp4")[0].split("-")[0][2:]) for vid in vids
    ]
    vids = [vids[i] for i in sorted(range(len(idx)), key=lambda k: idx[k])]
    idx = [idx[i] for i in sorted(range(len(idx)), key=lambda k: idx[k])]
    num = num or len(vids)

    frames = []
    for i, id in tqdm(enumerate(idx[:num]), total=num):
        # frames.append(imageio.mimread(vids[i])[i % 30])
        img = Image.fromarray(imageio.mimread(vids[i])[i % 30])
        # img = Image.open(self.get_save_path(filename))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), f"{id:5d}", (0, 0, 0))
        # img.save(self.get_save_path(filename))
        frames.append(img)

    imageio.mimwrite(os.path.join(path, "training.mp4"), frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument(
        "--num", type=int, default=None, help="path to image (png, jpeg, etc.)"
    )
    opt = parser.parse_args()

    # path = "/admin/home-vikram/git/threestudio/outputs/zero123/XL2_128_yellowduck_rgba.png@20230606-175353"
    run(opt.path, opt.num)
