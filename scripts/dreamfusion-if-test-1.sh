set -e
gpu=1
batch_size=4

prompts=("\"a 20-sided dice made out of glass\"" "lib:dslr_frazer" "lib:lion_head" "lib:dumplings" "lib:teapot")
# loop prompts
for prompt in "${prompts[@]}"
do
    echo "Prompt: $prompt"
    python launch.py --config configs/dreamfusion-if.yaml --train --gpu $gpu system.prompt_processor.prompt=$prompt data.batch_size=$batch_size
done
