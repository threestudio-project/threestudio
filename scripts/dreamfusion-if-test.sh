set -e
gpu=0
batch_size=4

python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt="a DSLR photo of a robot dinosaur" data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt="a 20-sided dice made out of glass" data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:croissant data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:snail data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:dslr_frazer data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:lion_head data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:dumplings data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:teapot data.batch_size=$batch_size
