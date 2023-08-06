gpu=0
cfg=outputs/att3d-if/composite_prompts@20230805-094945/configs/parsed.yaml
ckpt=outputs/att3d-if/composite_prompts@20230805-094945/ckpts/ckpt.ckpt
prompt_dir=load/composite_prompts_4.json

for ind in {0..15}
do
    echo $ind
    python launch.py --config $cfg --test --gpu $gpu resume=$ckpt data.n_test_views=1 system.prompt_processor.composite_prompt_dir=$prompt_dir system.prompt_processor.prompt_id=$ind
done

echo All done!
