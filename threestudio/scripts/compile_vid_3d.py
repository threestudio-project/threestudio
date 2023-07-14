import imageio
import numpy as np

vids = []
for i in range(3):
    vids.append([])
    for j in range(3):
        vid = imageio.get_reader(f"/Users/vikramvoleti/Downloads/vid{i+1}{j+1}.mp4")
        frames = [
            # np.concatenate([f[:, :128, :3], f[:, 384:512, :3]], axis=1) for f in vid
            f[:, :256, :3]
            for f in vid
        ]
        vids[-1].append(frames)

# # vids[0][0] = [vids[0][0][(k+50)%120] for k in range(120)]
# # vids[1][0] = [vids[1][0][(k+30)%120] for k in range(120)]
# vids[0][0] = [vids[0][0][-k][:, ::-1, :] for k in range(120)]
# vids[1][0] = [vids[1][0][-k][:, ::-1, :] for k in range(120)]
# vids[2][1] = [vids[2][1][(k+10)%120] for k in range(120)]
# vids[0][2] = [vids[0][2][(k+5)%120] for k in range(120)]
# vids[2][2] = [vids[2][2][(k+5)%120] for k in range(120)]

vid1 = []
for i in range(3):
    vid1.append(
        [
            np.concatenate([vids[i][0][k], vids[i][1][k], vids[i][2][k]], axis=1)
            for k in range(120)
        ]
    )

vid2 = [
    np.concatenate([vid1[0][k], vid1[1][k], vid1[2][k]], axis=0) for k in range(120)
]
imageio.mimwrite("/Users/vikramvoleti/Downloads/dognew12.mp4", vid2, fps=25)
