set -e
gpu=0
batch_size=4

prompts=(
    "lib:chimpanzee_banana"
    # "lib:chimpanzee_england_king"
)

seeds=(
    1
    2
    3
    4
)


for prompt in "${prompts[@]}"; do
    for seed in "${seeds[@]}"; do
        echo seed=$seed
        python launch.py --config configs/dreamfusion-if-perpneg.yaml --train --gpu $gpu system.prompt_processor.prompt="$prompt" data.batch_size=$batch_size data.n_val_views=2 system.loss.lambda_orient=[0,100.,10000.,5000] seed=$seed
    done
    # python launch.py --config configs/dreamfusion-if-perpneg.yaml --train --gpu $gpu system.prompt_processor.prompt="$prompt" data.batch_size=$batch_size data.n_val_views=2
done
