set -x
port=$(shuf -i25000-30000 -n1)
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
# export CUDA_VISIBLE_DEVICES='0,1'

#deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 src/train_lomo.py config/args_lomo.yaml
#deepspeed --master_port "$port" --include localhost:0 src/train_lomo.py config/args_lomo.yaml

# deepspeed --master_port "$port" --include localhost:0,1 src/train_lomo_bf16.py config/args_lomo_bf16.yaml
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 src/train_lomo_bf16.py config/args_lomo_bf16.yaml
#deepspeed --master_port "$port" --include localhost:0 src/train_lomo_lora.py config/args_lomo_lora.yaml
