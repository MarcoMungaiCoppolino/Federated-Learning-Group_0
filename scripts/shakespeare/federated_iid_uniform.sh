#!/bin/bash

wandb_key="$1"  # First argument is the Wandb key
wandb_username="$2"  # Second argument is the Wandb username

# Set other variables as needed
python_script="/content/Federated-Learning-Group_0/src/federated.py"
data_dir="/content/drive/MyDrive/MLDL/shakespeare/data/iid"
checkpoint_path="/content/drive/MyDrive/MLDL/shakespeare/checkpoints"
logfile="/content/drive/MyDrive/MLDL/shakespeare/logs/federated_shakespeare_100_iid_2000_uniform.log"
metrics_dir="/content/drive/MyDrive/MLDL/shakespeare/metrics"

# Check if GPU is available (using nvidia-smi command to check)
if nvidia-smi &> /dev/null; then
    gpu_arg="--gpu 0"
else
    gpu_arg=""
fi

# Execute the Python script with or without the Wandb key argument and GPU argument
if [ -n "$wandb_key" ]; then
    python3 "$python_script" \
        --dataset shakespeare \
        --model lstm \
        --iid 1 \
        --lr 1 \
        --epochs 200 \
        --data_dir "$data_dir" \
        --print_every 1 \
        $gpu_arg \
        --wandb_key "$wandb_key" \
        --wandb_username "$wandb_username" \
        --wandb_project Federated_Learning \
        --wandb_run_name "federated_shakespeare_100_iid_2000_uniform" \
        --local_ep 4 \
        --participation 1 \
        --logfile "$logfile" \
        --metrics_dir "$metrics_dir"
else
    python3 "$python_script" \
        --dataset shakespeare \
        --iid 1 \
        --print_every 1 \
        --lr 1 \
        --epochs 200 \
        --data_dir "$data_dir" \
        $gpu_arg \
        --local_ep 4 \
        --participation 1 \
        --logfile "$logfile" \
        --metrics_dir "$metrics_dir"
fi