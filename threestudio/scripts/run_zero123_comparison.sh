# with standard zero123
threestudio/scripts/run_zero123_phase.sh 6 anya_front 105000 0

# with zero123XL (not released yet!)
threestudio/scripts/run_zero123_phase.sh 1 anya_front XL_20230604 0
threestudio/scripts/run_zero123_phase.sh 2 baby_phoenix_on_ice XL_20230604 20
threestudio/scripts/run_zero123_phase.sh 3 beach_house_1 XL_20230604 50
threestudio/scripts/run_zero123_phase.sh 4 bollywood_actress XL_20230604 0
threestudio/scripts/run_zero123_phase.sh 5 beach_house_2 XL_20230604 30
threestudio/scripts/run_zero123_phase.sh 6 hamburger XL_20230604 10
threestudio/scripts/run_zero123_phase.sh 7 cactus XL_20230604 8
threestudio/scripts/run_zero123_phase.sh 0 catstatue XL_20230604 50
threestudio/scripts/run_zero123_phase.sh 1 church_ruins XL_20230604 0
threestudio/scripts/run_zero123_phase.sh 2 firekeeper XL_20230604 10
threestudio/scripts/run_zero123_phase.sh 3 futuristic_car XL_20230604 20
threestudio/scripts/run_zero123_phase.sh 4 mona_lisa XL_20230604 10
threestudio/scripts/run_zero123_phase.sh 5 teddy XL_20230604 20

# set guidance_eval to 0, to greatly speed up training
threestudio/scripts/run_zero123_phase.sh 7 anya_front XL_20230604 0 system.freq.guidance_eval=0

# disable wandb for faster training (or if you don't want to use it)
threestudio/scripts/run_zero123_phase.sh 7 anya_front XL_20230604 0 system.loggers.wandb.enable=false system.freq.guidance_eval=0
