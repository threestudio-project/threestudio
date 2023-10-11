import subprocess

prompt_list = [
    "a delicious hamburger",
    "A DSLR photo of a roast turkey on a platter",
    "A high quality photo of a dragon",
    "A DSLR photo of a bald eagle",
    "A bunch of blue rose, highly detailed",
    "A 3D model of an adorable cottage with a thatched roof",
    "A high quality photo of a furry corgi",
    "A DSLR photo of a panda",
    "a DSLR photo of a cat lying on its side batting at a ball of yarn",
    "a beautiful dress made out of fruit, on a mannequin. Studio lighting, high quality, high resolution",
    "a DSLR photo of a corgi wearing a beret and holding a baguette, standing up on two hind legs",
    "a zoomed out DSLR photo of a stack of pancakes",
    "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
]
negative_prompt = "oversaturated color, ugly, tiling, low quality, noise, ugly pattern"

gpu_id = 0
max_steps = 10
val_check = 1
out_name = "gsgen_baseline"
for prompt in prompt_list:
    print(f"Running model on device {gpu_id}: ", prompt)
    command = [
        "python", "launch.py",
        "--config", "configs/gaussian_splatting.yaml",
        "--train",
        f"system.prompt_processor.prompt={prompt}",
        f"system.prompt_processor.negative_prompt={negative_prompt}",
        f"name={out_name}",
        "--gpu", f"{gpu_id}"
    ]
    subprocess.run(command)
        