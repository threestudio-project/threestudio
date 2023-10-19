python launch.py --config configs/gaussian_splatting.yaml --train --gpu 0 name=gaussian_splatting tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"

# python launch.py --config configs/gaussian_splatting.yaml --train --gpu 0 name=gaussian_splatting tag=v1 system.prompt_processor.prompt="a delicious hamburger"
python launch.py --config configs/gaussian_splatting_editing.yaml --train --gpu 0 data.dataroot="load/twindom" name=gaussian_splatting_editing tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"

# gaussian
python launch.py --config configs/gaussian_dynamic.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" data.eval_interpolation=[0,20,10] data.eval_time_interpolation=[0.0,0.8]

python launch.py --config configs/gaussian_dynamic.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" data.eval_interpolation=[0,20,10] data.eval_time_interpolation=[0.0,0.8] resume=/root/autodl-tmp/threestudio/outputs/dynamic_gaussian/v1@20231015-221401/ckpts/last.ckpt

python launch.py --config configs/gaussian_dynamic.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" data.eval_interpolation=[0,3,40] data.eval_time_interpolation=[0.0,0.8] resume=/root/autodl-tmp/threestudio/outputs/dynamic_gaussian/v1@20231016-194337/ckpts/epoch=0-step=50000.ckpt

python launch.py --config configs/dynamic_nerf.yaml --train --gpu 0  data.dataroot="load/twindom_dynamic" name=dynamic_nerf tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" data.eval_interpolation=[0,20,10] data.eval_time_interpolation=[0.0,0.8]

python launch.py --config configs/dynamic_control4d.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_control4d system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" data.eval_interpolation=[0,3,40] data.eval_time_interpolation=[0.0,0.8]

python launch.py --config configs/dynamic_gaussian_instruct.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian tag=v1 system.prompt_processor.prompt="Turn him into Elon Musk"
python launch.py --config configs/dynamic_gaussian_instruct.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian tag=v1 data.camera_layout="front" data.camera_distance=1.0 system.prompt_processor.prompt="Elon Musk wearing white shirt" data.eval_interpolation=[0,1,40] data.eval_time_interpolation=[0.0,0.8]

python launch.py --config configs/dynamic_gaussian_reconstruct.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian_reconstruct tag=v1 data.camera_layout="front" data.camera_distance=1.0 system.prompt_processor.prompt="Elon Musk wearing white shirt" data.eval_interpolation=[10,11,40] data.eval_time_interpolation=[0.0,0.8]
python launch.py --config configs/dynamic_gaussian_instruct.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian_instruct tag=v1 data.camera_layout="front" data.camera_distance=1.0 system.prompt_processor.prompt="Elon Musk wearing white shirt" data.eval_interpolation=[10,11,40] data.eval_time_interpolation=[0.0,0.8] resume=/root/autodl-tmp/threestudio/outputs/dynamic_gaussian_reconstruct/v1@20231019-214518/ckpts/last.ckpt
python launch.py --config configs/dynamic_gaussian_volume_instruct.yaml --train --gpu 0 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian_instruct tag=v1 data.camera_layout="front" data.camera_distance=1.0 system.prompt_processor.prompt="Elon Musk wearing white shirt" data.eval_interpolation=[10,11,40] data.eval_time_interpolation=[0.0,0.8] system.geometry.geometry_convert_from=/root/autodl-tmp/threestudio/outputs/dynamic_gaussian_reconstruct/v1@20231019-214518/ckpts/last.ckpt
