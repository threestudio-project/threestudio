import base64
import logging
from io import BytesIO
from pydantic import BaseModel, Field
from enum import Enum

from PIL import Image


def pad_and_resize_image(im, output_path):
    TARGET_WIDTH = 512
    TARGET_HEIGHT = 512
    shrink_ratio = max((im.size[0]) / TARGET_WIDTH, (im.size[1]) / TARGET_HEIGHT)
    print(shrink_ratio, im.size[0])

    im = im.resize((int(im.size[0] / shrink_ratio), int(im.size[1] / shrink_ratio)))
    fill_color = im.getpixel((0, 0))
    print(im.size)
    output_image = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), fill_color)
    output_image.paste(
        im, ((TARGET_WIDTH - im.size[0]) // 2, (TARGET_HEIGHT - im.size[1]) // 2)
    )

    output_image.save(output_path)


def image_from_b64(image: str) -> Image.Image:
    try:
        return Image.open(BytesIO(base64.b64decode(image)))
    except Exception as e:
        logging.error(f"Invalid base64 image: {e}")
        raise ValueError(f"Invalid image")


PHASE1_CONFIGS = [
    "zero123_sai_multinoise_amb.yaml",
    "zero123_sai_multinoise_amb_new.yaml",
    "zero123_sai_multinoise_amb_raysdivisor.yaml",
    "zero123_sai_multinoise_amb_fast.yaml",
    "zero123.yaml",
    "zero123_64.yaml",
    "zero123_3sai.yaml",
    "zero123_2xl.yaml",
    "zero123_1original.yaml",
]

PHASE2_CONFIGS = ["zero123_magic123refine.yaml", "zero123_magic123refine_new.yaml"]


class Request(BaseModel):
    prompt: str
    init_image: str
    seed: int = 0
    coarse_steps: int = 600
    refine_steps: int = 1000
    elevation_angle: float = 5
    azimuth_angle: float = 0
    phase1_config: str = PHASE1_CONFIGS[0]
    phase2_config: str = PHASE2_CONFIGS[0]


class FinishReason(str, Enum):
    SUCCESS = "SUCCESS"
    FILTER = "CONTENT_FILTERED"


class Response(BaseModel):
    model: str
    texture: str
    material: str
    finish_reason: FinishReason
    seed: int
