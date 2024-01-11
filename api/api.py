import base64
import glob
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from multiprocessing import Process
from typing import List
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, Response
from PIL import Image
from pydantic import BaseModel, Field

MODELS_PATH = os.getcwd() + "/../database/models/"
DATABASE_PATH = os.getcwd() + "/../database/"
THREESTUDIO_PATH = os.getcwd() + "/../threestudio/"


logger = logging.getLogger(__name__)


def _image_from_b64(image: str) -> Image.Image:
    try:
        return Image.open(BytesIO(base64.b64decode(image)))
    except Exception as e:
        logging.error(f"Invalid base64 image: {e}")
        raise ValueError(f"Invalid image")


CONTENT_TYPE_JSON = "application/json"


# todo optional values
class Request(BaseModel):
    user_id: str = Field("")
    prompt: str = Field("")
    init_image: str = Field(None)
    seed: int = Field(0)
    coarse_steps: int = Field(600)
    refine_steps: int = Field(1000)
    elevation_angle: float = Field(5)
    azimuth_angle: float = Field(0)
    phase1_config: str = Field("")
    phase2_config: str = Field("")


class FinishReason(str, Enum):
    SUCCESS = "SUCCESS"
    FILTER = "CONTENT_FILTERED"


@dataclass
class CreateModelResponse:
    model: str
    texture: str
    material: str
    finish_reason: FinishReason
    seed: int


def wait_for_model(
    model_id,
    threestudio_output_path,
    image_path,
    obj_path,
    tex_path,
    mtl_path,
    error_path,
    user_id,
    coarse_steps,
    refine_steps,
):
    global processes
    print(
        "Waiting for model ",
        model_id,
        obj_path,
        tex_path,
        mtl_path,
        image_path,
        error_path,
    )

    start = time.time()
    TIMEOUT = 4 * 60 * 60
    while (
        time.time() - start < TIMEOUT
        and not os.path.isfile(obj_path)
        and not os.path.isfile(error_path)
    ):
        time.sleep(120)

    model_folder = MODELS_PATH + str(user_id) + str(model_id) + "/"
    os.makedirs(model_folder, exist_ok=True)

    if os.path.isfile(error_path):
        try:
            f = open(error_path, "r")
            msg = f.read()
            f.close()
            os.rename(image_path, model_folder + "input.png")
            os.rename(error_path, model_folder + "error.txt")
            shutil.rmtree(threestudio_output_path)
        except:
            print("Error case: Could not copy intermediate results and cleanup:", e)
        return

    try:
        os.rename(image_path, model_folder + "input.png")
        os.rename(obj_path, model_folder + "model.obj")
        os.rename(tex_path, model_folder + "texture_kd.jpg")
        os.rename(mtl_path, model_folder + "model.mtl")
    except Exception as e:
        print("When copying outputs: ", e)
        return

    try:
        os.rename(
            threestudio_output_path + f"/Phase2/save/it{refine_steps}-test.mp4",
            model_folder + "Phase2.mp4",
        )
        os.rename(
            threestudio_output_path + f"/Phase1/save/it{coarse_steps}-test.mp4",
            model_folder + "Phase1.mp4",
        )
        shutil.rmtree(threestudio_output_path)
    except Exception as e:
        print("Could not copy intermediate results and cleanup:", e)

    print(f"updated model {model_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(THREESTUDIO_PATH + "/outputs/*")
    for threestudio_output_path in glob.glob(THREESTUDIO_PATH + "/outputs/*"):
        parts = os.path.basename(threestudio_output_path).split("_")
        if len(parts) == 4:
            try:
                assert len(parts) == 4
                model_id, user_id, coarse_steps, refine_steps = parts

                input_image_path = threestudio_output_path + "/input.png"
                phase3_threestudio_output_path = (
                    threestudio_output_path + "/Phase3/save/it0-export/"
                )

                obj_path = phase3_threestudio_output_path + "/model.obj"
                texture_path = phase3_threestudio_output_path + "/texture_kd.jpg"
                mtl_path = phase3_threestudio_output_path + "/model.mtl"
                error_path = threestudio_output_path + "/error.txt"
                Process(
                    target=wait_for_model,
                    args=(
                        model_id,
                        threestudio_output_path,
                        input_image_path,
                        obj_path,
                        texture_path,
                        mtl_path,
                        error_path,
                        user_id,
                        int(coarse_steps),
                        int(refine_steps),
                    ),
                ).start()
            except Exception as e:
                print(f"Could not launch process for {threestudio_output_path} {e}")
    yield
    # shutdown


app = FastAPI(lifespan=lifespan)


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


@app.post("/image-to-model/")
def create_model(request: Request, response: Response):
    if (len(request.user_id)) == 0:
        response.status_code = 422
        return {"name": "bad_request", "message": "user required"}

    if len(request.prompt) == 0:
        response.status_code = 422
        return {"name": "bad_request", "message": "prompt required"}

    if len(request.prompt) >= 256:
        response.status_code = 422
        return {
            "name": "bad_request",
            "message": "prompt must be less than 256 characters long",
        }

    job_folder = (
        f"{model_id}_{request.user_id}_{request.coarse_steps}_{request.refine_steps}"
    )
    threestudio_output_path = THREESTUDIO_PATH + "/outputs/" + job_folder + "/"

    input_image_path = threestudio_output_path + "input.png"

    os.makedirs(os.path.dirname(threestudio_output_path), exist_ok=True)

    png_image = _image_from_b64(request.init_image)
    pad_and_resize_image(png_image, input_image_path)

    phase3_threestudio_output_path = threestudio_output_path + "Phase3/save/it0-export/"

    obj_path = phase3_threestudio_output_path + "model.obj"
    texture_path = phase3_threestudio_output_path + "texture_kd.jpg"
    mtl_path = phase3_threestudio_output_path + "model.mtl"
    error_path = threestudio_output_path + "error.txt"

    if os.path.isfile(obj_path):
        os.remove(obj_path)

    if os.path.isfile(error_path):
        os.remove(error_path)

    print(
        f'cd .. && sbatch run_request.sh {input_image_path} {job_folder} "{request.prompt}" {request.elevation_angle} {request.coarse_steps} {request.refine_steps} {request.azimuth_angle} {request.phase1_config} {request.phase2_config}'
    )
    os.system(
        f'cd .. && sbatch run_request.sh {input_image_path} {job_folder} "{request.prompt}" {request.elevation_angle} {request.coarse_steps} {request.refine_steps} {request.azimuth_angle} {request.phase1_config} {request.phase2_config}'
    )

    Process(
        target=wait_for_model,
        args=(
            model_id,
            threestudio_output_path,
            input_image_path,
            obj_path,
            texture_path,
            mtl_path,
            error_path,
            request.user_id,
            request.coarse_steps,
            request.refine_steps,
        ),
    ).start()

    print(f"Request succeeded {request.prompt}")

    return {"message": "launched"}


@dataclass
class Preview:
    image: str
    prompt: str
    user_name: str
    num_downloads: str


@dataclass
class PreviewRecentResponse:
    recents: List[Preview]


@app.get("/get-model/")
def get_model(index: int, up_axis: str, forward: str, response: Response):
    forward = forward.strip()
    up_axis = up_axis.strip()
    if up_axis not in set(["x", "y", "z"]):
        response.status_code = 422
        return {"name": "bad_request", "message": "up_axis must be one of x,y,z"}

    ret = ""
    paths = []

    if paths[0] == None:
        return ret

    tex_file_path, model_file_path, mtl = paths

    base_path = os.path.dirname(model_file_path)

    out_glb_path = base_path + "/model.glb"
    out_obj_path = base_path + "/model.obj"

    t = m.export(out_glb_path)
    t = m.export(out_obj_path)

    zip_io = BytesIO()
    with ZipFile(zip_io, "w", compression=ZIP_DEFLATED) as z:
        zip_tex_path = os.path.basename(paths[0])  # e.g. texture_kd.jpg
        zip_obj_path = os.path.basename(paths[1])

        z.write(tex_file_path, zip_tex_path)
        z.write(out_obj_path, "model.obj")
        z.write(out_glb_path, "model.glb")
        z.writestr(zip_obj_path[:-4] + ".mtl", mtl)

        preview_path = os.path.dirname(model_file_path) + "/Phase2.mp4"
        if os.path.isfile(preview_path):
            z.write(preview_path, "preview.mp4")
        input_image_path = os.path.dirname(model_file_path) + "/input.png"
        if os.path.isfile(input_image_path):
            z.write(input_image_path, "input.png")

    return base64.b64encode(zip_io.getvalue()).decode("utf-8")


@app.get("/get-turntable")
def get_turntable(index: int):
    vid_data_b64 = ""

    if pathes[0] == None:
        pass
    else:
        model_file_path = pathes[0]
        preview_path = os.path.dirname(model_file_path) + "/Phase2.mp4"
        if os.path.isfile(preview_path):
            with open(preview_path, "rb") as f:
                vid_data_b64 = base64.b64encode(f.read())

    return vid_data_b64
