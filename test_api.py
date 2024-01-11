import logging
from os.path import join
from pathlib import Path
from api.utils import Request, Response
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

DEFAULT_PROMPT = "a painting of a cat"

test_assets_dir = join(Path(__file__).parent.parent.resolve(), "assets")
output_dir = join(Path(__file__).parent.resolve(), "samples")


def run_request(params: Request, init_image_path: str, api_url: str) -> Response:
    init_image_bytes: bytes = open(init_image_path, "rb").read()
    params.init_image = init_image_bytes.decode("utf-8")

    response = requests.post(
        api_url,
    )

    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    response_dict = json.loads(response.text)
    response = Response(**response_dict)
    return response


run_request(
    Request(prompt="A doll", init_image="load/images/anya_front.png"),
    join(test_assets_dir, "cat.jpg"),
    "http://localhost:8080/invocations",
)
