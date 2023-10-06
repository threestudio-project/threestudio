python launch.py --config configs/gaussian_splatting.yaml --train --gpu 0 name=gaussian_splatting tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"

# python launch.py --config configs/gaussian_splatting.yaml --train --gpu 0 name=gaussian_splatting tag=v1 system.prompt_processor.prompt="a delicious hamburger"
python launch.py --config configs/gaussian_splatting_editing.yaml --train --gpu 0 data.dataroot="load/twindom" name=gaussian_splatting_editing tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"

python launch.py --config configs/gaussian_dynamic.yaml --train --gpu 1 data.dataroot="load/twindom_dynamic" name=dynamic_gaussian tag=v1 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" data.eval_interpolation=[0,20,10] data.eval_time_interpolation=[0.0,0.8]

