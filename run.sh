gpu=2
cfg=outputs/att3d-if/composite_prompts@20230805-165224/configs/parsed.yaml
ckpt=outputs/att3d-if/composite_prompts@20230805-165224/ckpts/last.ckpt
prompt_dir=load/DF27.json

for ind in {0..26}
do
    echo $ind
    python launch.py --config $cfg --test --gpu $gpu resume=$ckpt system.prompt_processor.composite_prompt_dir=$prompt_dir system.prompt_processor.prompt_id=$ind
done

echo All done!
