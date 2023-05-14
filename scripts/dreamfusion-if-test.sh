set -e
gpu=0
batch_size=4

prompts=("lib:eyeglasses" "\"a DSLR photo of a robot dinosaur\"" "lib:croissant" "lib:snail")
# loop prompts
for prompt in "${prompts[@]}"
do
    echo "Prompt: $prompt"
    python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=$prompt data.batch_size=$batch_size
done
