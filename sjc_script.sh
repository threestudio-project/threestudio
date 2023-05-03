# trump
python launch.py --train --config configs/sjc.yaml --gpu 0 tag=sjc-trump system.prompt_processor.prompt="Trump figure" trainer.max_steps=30000 system.loss.lambda_emptiness=[15000,1.e+4,2.e+5,15001] seed=1

# obama
python launch.py --train --config configs/sjc.yaml --gpu 1 tag=sjc-obama system.prompt_processor.prompt="Obama figure" trainer.max_steps=30000 system.loss.lambda_emptiness=[15000,1.e+4,2.e+5,15001] system.guidance.var_red=False

# biden
python launch.py --train --config configs/sjc.yaml --gpu 1 tag=sjc-biden system.prompt_processor.prompt="Biden figure" 

# temple of heaven
python launch.py --train --config configs/sjc.yaml --gpu 0 tag=sjc-temple system.prompt_processor.prompt="A zoomed out high quality photo of Temple of Heaven" 

# burger
python launch.py --train --config configs/sjc.yaml --gpu 1 tag=sjc-burger system.prompt_processor.prompt="A high quality photo of a delicious burger"
 
# rocket
python launch.py --train --config configs/sjc.yaml --gpu 1 tag=sjc-rocket system.prompt_processor.prompt="A wide angle zoomed out photo of Saturn V rocket from distance" trainer.max_steps=30000 system.loss.lambda_emptiness=[15000,1.e+4,2.e+5,15001] system.guidance.var_red=False

# tank
python launch.py --train --config configs/sjc.yaml --gpu 0 tag=sjc-tank system.prompt_processor.prompt="A product photo of a toy tank" trainer.max_steps=20000 system.loss.lambda_emptiness=[10000,1.e+4,2.e+5,10001]

# horse
python launch.py --train --config configs/sjc.yaml --gpu 0 tag=sjc-horse system.prompt_processor.prompt="A photo of a horse walking"

# duck
python launch.py --train --config configs/sjc.yaml --gpu 0 tag=sjc-duck system.prompt_processor.prompt="a DSLR photo of a yellow duck" system.loss.lambda_depth=10.0

# chair
python launch.py --train --config configs/sjc.yaml --gpu 0 tag=sjc-chair system.optimizer.args.lr=0.01 system.prompt_processor.prompt="A high quality photo of a Victorian style wooden chair with velvet upholstery" trainer.max_steps=50000 system.loss.lambda_emptiness=[25000,7000.,14000.,25001] system.loss.lambda_depth=10.0

# school bus
python launch.py --train --config configs/sjc.yaml --gpu 0 tag=sjc-bus system.prompt_processor.prompt="A high quality photo of a yellow school bus" trainer.max_steps=30000 system.loss.lambda_emptiness=[15000,1.e+4,2.e+5,15001] system.guidance.var_red=False
