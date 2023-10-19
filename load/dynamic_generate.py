import json
import math
import os

import numpy as np


def rotationY(theta):
    return np.array(
        [
            [math.cos(theta), 0, -math.sin(theta), 0],
            [0, 1, 0, 0],
            [math.sin(theta), 0, math.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )


os.makedirs("twindom_dynamic", exist_ok=True)

config = json.load(open(os.path.join("twindom", "transforms.json")))
config = dict(config)

new_config = config.copy()
new_config["frames"] = []

view_list = list(range(80, 90, 1)) + list(range(0, 10, 1))
for _ in range(0, 30):
    if _ == 0:
        K = 20
    else:
        K = 4
    for __ in range(0, K):
        if _ == 0:
            frame_i = view_list[__]
            t = 0
        else:
            frame_i = view_list[__ * 5]
            t = np.random.randint(0, 1000) / 1000
        print(frame_i)
        frame = config["frames"][frame_i]
        new_frame = {}
        key_value_list = ["fl_x", "fl_y", "cx", "cy", "w", "h", "file_path"]
        for key in key_value_list:
            new_frame[key] = frame[key]
        new_frame["moment"] = t
        matrix = [[frame["transform_matrix"][i][j] for j in range(4)] for i in range(4)]
        c2w = np.array(matrix)
        w2c = np.linalg.inv(c2w)
        w2c = w2c @ rotationY(t * math.acos(-1) / 4)
        c2w = np.linalg.inv(w2c)
        # matrix[1][3] += 0.5 * math.sin(t * math.acos(-1) / 2)
        new_frame["transform_matrix"] = [
            [c2w[i][j] for j in range(4)] for i in range(4)
        ]
        new_config["frames"].append(new_frame)
        # print(new_config["frames"])
        # exit(0)

# for _ in range(0, 15):
#     for __ in range(0, 10):
#         if _ == 0:
#             frame_i = __ * 9
#             t = 0
#         else:
#             frame_i = np.random.randint(0, 90)
#             t = np.random.randint(0, 1000) / 1000
#         frame = config["frames"][frame_i]
#         new_frame = {}
#         key_value_list = ["fl_x", "fl_y", "cx", "cy", "w", "h", "file_path"]
#         for key in key_value_list:
#             new_frame[key] = frame[key]
#         new_frame["moment"] = t
#         matrix = [[frame["transform_matrix"][i][j] for j in range(4)] for i in range(4)]
#         matrix[1][3] += 0.5 * t
#         new_frame["transform_matrix"] = matrix.copy()
#         new_config["frames"].append(new_frame)
#         # print(new_config["frames"])
#         # exit(0)
print(new_config["frames"][0])
json.dump(
    new_config, open(os.path.join("twindom_dynamic", "transforms.json"), "w"), indent=4
)
