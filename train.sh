export OPENAI_LOGDIR=./logs/

python main.py -t -n experimental --base ./configs/stable-diffusion/v1-finetune.yaml --no-test --seed 25 --scale_lr False --data_root "./dataset" --gpus=1