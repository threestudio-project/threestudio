import json
import os
import numpy as np
os.makedirs('twindom_dynamic', exist_ok=True)

config = json.load(open(os.path.join('twindom', 'transforms.json')))
config = dict(config)

new_config = config.copy()
new_config["frames"] = []
    
for _ in range(0, 12):
    for __ in range(0, 15):
        if _ == 0:
            frame_i = __ * 6
            t = 0
        else:
            frame_i = np.random.randint(0, 90)
            t = np.random.randint(0, 1000) / 1000
        frame = config["frames"][frame_i]
        new_frame = frame.copy()
        new_frame["moment"] = t
        matrix = [[new_frame["transform_matrix"][i][j] for j in range(4) ] for i in range(4)]
        matrix[1][3] += 0.5*t
        new_frame["transform_matrix"] = matrix.copy()
        new_config["frames"].append(new_frame)
        # print(new_config["frames"])
        # exit(0)
print(new_config["frames"][0])
json.dump(new_config, open(os.path.join('twindom_dynamic', 'transforms.json'), 'w'), indent=4)


