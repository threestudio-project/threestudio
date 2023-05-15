set -e
gpu=1
batch_size=4

python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:eiffel data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:teddy_selfie data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:frosting data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:anvil data.batch_size=$batch_size
python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=lib:liberty data.batch_size=$batch_size
