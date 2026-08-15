# fixed_params
torchrun --standalone --nproc_per_node=2 train.py --outdir=saved_models/fixed_params --data=Data/fixed_params/train_data64 --stats Stats.yaml --num_classes 3 --exp fixed_params --arch=ddpmpp --batch=64 --batch-gpu=32 --cond=1 --tick=20 --snap=20 --dump=20 --duration=20 --ema=0.05

# var_params
torchrun --standalone --nproc_per_node=2 train.py --outdir=saved_models/var_params --data=Data/var_params/train_data64 --stats Stats.yaml --num_classes 3 --exp var_params --arch=ddpmpp --batch=64 --batch-gpu=32 --cond=1 --tick=20 --snap=20 --dump=20 --duration=20 --ema=0.05

# hetro_params
torchrun --standalone --nproc_per_node=2 train.py --outdir=saved_models/hetro_params --data=Data/hetro_params/train_data64 --stats Stats.yaml --num_classes 3 --exp hetro_params --arch=ddpmpp --batch=64 --batch-gpu=32 --cond=1 --tick=20 --snap=20 --dump=20 --duration=40 --ema=0.05

# hetro_params_physics
torchrun --standalone --nproc_per_node=2 train.py --outdir=saved_models/hetro_params_physics --data=Data/hetro_params/train_data64 --stats Stats.yaml --num_classes 3 --exp hetro_params_physics --arch=ddpmpp --batch=64 --batch-gpu=32 --cond=1 --tick=20 --snap=20 --dump=20 --duration=40 --ema=0.05

# scalar_vector
torchrun --standalone --nproc_per_node=2 train.py --outdir=saved_models/scalar_vector --data=Data/scalar_vector/train_data64 --stats Stats.yaml --num_classes 9 --exp scalar_vector --arch=ddpmpp --batch=64 --batch-gpu=32 --cond=1 --tick=20 --snap=20 --dump=20 --duration=40 --ema=0.05
