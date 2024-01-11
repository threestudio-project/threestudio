import base64
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import tempfile
import uvicorn
from fastapi import FastAPI, Response

from api.threestudio_runner import launch_threestudio
from api.utils import *


CONTENT_TYPE_JSON = "application/json"

# copied from ComfyUI_stability/servers/triton/api_server.py
log_format = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger(__name__)

# TODO add warmup request
app = FastAPI()


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


@app.get("/ping")
async def ping():
    return {"status": "healthy"}


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    
    # TODO prompt validation

    model_id = int(time.time() * 1000)
    job_folder = (
        f"{model_id}_{request.user_id}_{request.coarse_steps}_{request.refine_steps}"
    )
    threestudio_output_path = "../outputs/" + job_folder + "/"

    input_image_path = threestudio_output_path + "input.png"

    os.makedirs(os.path.dirname(threestudio_output_path), exist_ok=True)

    png_image = image_from_b64(request.init_image)
    pad_and_resize_image(png_image, input_image_path)

    phase3_threestudio_output_path = threestudio_output_path + "Phase3/save/it0-export/"

    model_folder = tempfile.mkdtemp(prefix=f"threestudio_model_{model_id}_")
    launch_threestudio(request, model_folder)   
    export_package(phase3_threestudio_output_path, "y", "z")

    # TODO clean up / remove tmpdir

    # TODO parse response

    return Response()


def export_package(output_path: str, up_axis: str, forward: str):
    forward = forward.strip()
    up_axis = up_axis.strip()
    if up_axis not in set(["x", "y", "z"]):
        return {"name": "bad_request", "message": "up_axis must be one of x,y,z"}

    ret = ""
    paths = []

    if paths[0] == None:
        return ret

    obj_path = output_path + "model.obj"
    texture_path = output_path + "texture_kd.jpg"
    mtl_path = output_path + "model.mtl"
    error_path = output_path + "error.txt"
    glb_path = output_path + "model.glb"

    # TODO read meshes with trimesh

    zip_io = BytesIO()
    with ZipFile(zip_io, "w", compression=ZIP_DEFLATED) as z:
        zip_tex_path = os.path.basename(paths[0])  # e.g. texture_kd.jpg
        zip_obj_path = os.path.basename(paths[1])

        z.write(texture_path, zip_tex_path)
        z.write(obj_path, "model.obj")
        z.write(glb_path, "model.glb")
        z.writestr(zip_obj_path[:-4] + ".mtl", mtl_path)

        # TODO fix
        # preview_path = os.path.dirname(model_file_path) + "/Phase2.mp4"
        # if os.path.isfile(preview_path):
        #     z.write(preview_path, "preview.mp4")
        # input_image_path = os.path.dirname(model_file_path) + "/input.png"
        # if os.path.isfile(input_image_path):
        #     z.write(input_image_path, "input.png")

    return base64.b64encode(zip_io.getvalue()).decode("utf-8")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)