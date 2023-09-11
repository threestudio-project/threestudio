# magic123
CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/magic123-coarse-sd.yaml --train data.image_path=load/images/luffy_medal_rgba.png system.prompt_processor.prompt="a high-resolution DSLR image of Luffy medal" 
CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/magic123-refine-sd.yaml --train data.image_path=load/images/luffy_medal_rgba.png system.prompt_processor.prompt="a high-resolution DSLR image of Luffy medal" system.geometry_convert_from=outputs/magic123-coarse-sd/luffy_medal_rgba.png-a_high-resolution_DSLR_image_of_Luffy_medal@20230901-234317/ckpts/last.ckpt system.renderer.context_type=cuda

# zero123
CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/zero123.yaml --train data.image_path=load/images/luffy_medal_rgba.png system.guidance.pretrained_model_name_or_path=/home/asset/threestudio/load/zero123/zero123-xl.ckpt

#prolific dreamer
CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/prolificdreamer-patch.yaml --train system.prompt_processor.prompt="a high-resolution DSLR image of Luffy medal"
CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/prolificdreamer-patch.yaml --train system.prompt_processor.prompt="a high-resolution DSLR image of gold circular Luffy medal"
CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/prolificdreamer-patch.yaml --train system.prompt_processor.prompt="a high-resolution DSLR image of gold flat thin Luffy medal"

#magic123 with VSD
CUDA_VISIBLE_DEVICES=1 python launch.py --config configs/magic123-coarse-sd_VSD.yaml --train data.image_path=load/images/luffy_medal_rgba.png system.prompt_processor.prompt="a high-resolution DSLR image of Luffy medal" 
