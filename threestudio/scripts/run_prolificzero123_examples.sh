python launch.py --config configs/prolificzero123.yaml --train --gpu 0 data.image_path=./load/images/lego_batman_rgba.png system.prompt_processor.prompt="A high quality 3D render of lego batman" system.freq.guidance_eval=13 system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-prolificzero123"

python launch.py --config configs/prolificzero123.yaml --train --gpu 3 data.image_path=./load/images/anya_front_rgba.png system.prompt_processor.prompt="A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" system.freq.guidance_eval=13 system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-prolificzero123"



# claforte tests
# --------------------
#
python launch.py --config configs/prolificzero123.yaml --train --gpu 0 \
  data.image_path=./load/images/anya_front_rgba.png \
  system.prompt_processor.prompt="A high quality 3D render of a kindergarden anime girl 5 years old with pink hair, standing up proudly with her hands in the air" \
  system.freq.guidance_eval=13 system.loggers.wandb.enable=true \
  system.loggers.wandb.project="claforte-prolificzero123" system.loggers.wandb.name="anya_vikram"

# change settings
python launch.py --config configs/prolificzero123.yaml --train --gpu 0 \
  data.image_path=./load/images/anya_front_rgba.png \
  system.prompt_processor.prompt="A high quality 3D render of a kindergarden anime girl 5 years old with pink hair, standing up proudly with her hands in the air" \
  system.freq.guidance_eval=13 system.loggers.wandb.enable=true \
  system.loggers.wandb.project="claforte-prolificzero123" system.loggers.wandb.name="anya_enable_ref_loss"
