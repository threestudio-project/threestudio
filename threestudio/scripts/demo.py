# 1. Generate using StableDiffusionXL https://clipdrop.co/stable-diffusion

# 2. Remove background https://clipdrop.co/remove-background

# 3. Estimate depth and normal https://omnidata.vision/demo/ (I used Omnidata Normal (with X-TC & 3DCC), and MiDaS Depth)


# 4. Convert depth image from RGB to greyscale
def depth_rgb_to_grey(depth_filename):
    # depth_filename = "image_depth.png"
    import shutil

    import cv2
    import numpy as np

    shutil.copyfile(depth_filename, depth_filename.replace("_depth", "_depth_orig"))
    depth = cv2.imread(depth_filename)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    mask = (
        cv2.resize(
            cv2.imread(depth_filename.replace("_depth", "_rgba"), cv2.IMREAD_UNCHANGED)[
                :, :, -1
            ],
            depth.shape,
        )
        > 0
    )
    # depth[mask] = (depth[mask] - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth[~mask] = 0
    depth = (depth * 255).astype(np.uint8)
    cv2.imwrite(depth_filename, depth)


# 4. Mask normal
def normal_mask(normal_filename):
    # filename = "image_normal.png"
    import shutil

    import cv2

    shutil.copyfile(normal_filename, normal_filename.replace("_normal", "_normal_orig"))
    normal = cv2.imread(normal_filename)
    mask = (
        cv2.resize(
            cv2.imread(
                normal_filename.replace("_normal", "_rgba"), cv2.IMREAD_UNCHANGED
            )[:, :, -1],
            normal.shape[:2],
        )
        > 0
    )
    normal[~mask] = 0
    cv2.imwrite(normal_filename, normal)


# 5. Run Zero123
