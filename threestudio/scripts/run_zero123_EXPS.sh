NAME="dragon2"

# (400it) 64
# Amb_ratio 0.1 + 0.9*rand
# Bg_color always white
python launch.py --config configs/zero123.yaml --train --gpu 0 data.image_path=./load/images/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_EXPS tag="Phase1_64_amb0.1_bg100" system.amb_ratio_min=0.1 system.background.random_aug=False
# Bg_color 80% white
python launch.py --config configs/zero123.yaml --train --gpu 1 data.image_path=./load/images/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_EXPS tag="Phase1_64_amb0.1_bg80" system.amb_ratio_min=0.1 system.background.random_aug=True system.background.random_aug_prob=0.2
# Bg_color 50% white
python launch.py --config configs/zero123.yaml --train --gpu 2 data.image_path=./load/images/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_EXPS tag="Phase1_64_amb0.1_bg50" system.amb_ratio_min=0.1 system.background.random_aug=True system.background.random_aug_prob=0.5
# Amb_ratio 0.5 + 0.5*rand
# Bg_color always white
python launch.py --config configs/zero123.yaml --train --gpu 3 data.image_path=./load/images/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_EXPS tag="Phase1_64_amb0.5_bg100" system.amb_ratio_min=0.5 system.background.random_aug=False
# Bg_color 80% white
python launch.py --config configs/zero123.yaml --train --gpu 4 data.image_path=./load/images/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_EXPS tag="Phase1_64_amb0.5_bg80" system.amb_ratio_min=0.5 system.background.random_aug=True system.background.random_aug_prob=0.2
# Bg_color 50% white
python launch.py --config configs/zero123.yaml --train --gpu 5 data.image_path=./load/images/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_EXPS tag="Phase1_64_amb0.5_bg50" system.amb_ratio_min=0.5 system.background.random_aug=True system.background.random_aug_prob=0.5

# Amb_ratio 0.5 + 0.5*rand
# Bg_color always white
# (400it) 64
# DONE in GPU0
# (200it)64, (200it) 128
python launch.py --config configs/zero123_64_128.yaml --train --gpu 6 data.image_path=./load/images/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_EXPS tag="Phase1_64_128_amb0.5_bg100"
# (200it) 64, (100it) 128, (100it) 256
python launch.py --config configs/zero123.yaml --train --gpu 7 data.image_path=./load/images/${NAME}_rgba.png system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" use_timestamp=False name=${NAME}_EXPS tag="Phase1_64_128_256_amb0.5_bg100"
