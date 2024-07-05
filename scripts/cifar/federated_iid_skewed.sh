#!/bin/bash

wandb_key="$1"  # First argument is the Wandb key
wandb_username="$2"  # Second argument is the Wandb username

# Set other variables as needed
python_script="/content/Federated-Learning-Group_0/src/federated.py"
data_dir="/content/drive/MyDrive/MLDL/cifar/data"
checkpoint_path="/content/drive/MyDrive/MLDL/cifar/checkpoints"
logfile_base="/content/drive/MyDrive/MLDL/cifar/logs/federated_cifar_100_iid_2000_skewed"
metrics_dir="/content/drive/MyDrive/MLDL/cifar/metrics"

# Check if GPU is available (using nvidia-smi command to check)
if nvidia-smi &> /dev/null; then
    gpu_arg="--gpu 0"
else
    gpu_arg=""
fi

# Array of gamma values
gammas=(0.1 0.5 0.7)

# Loop through gamma values and execute the Python script for each
for gamma in "${gammas[@]}"; do
    logfile="${logfile_base}_gamma_${gamma}.log"
    
    if [ -n "$wandb_key" ]; then
        python3 "$python_script" \
            --dataset cifar \
            --iid 1 \
            --epochs 2000 \
            --data_dir "$data_dir" \
            $gpu_arg \
            --wandb_key "$wandb_key" \
            --wandb_username "$wandb_username" \
            --wandb_project Federated_Learning \
            --wandb_run_name "federated_cifar_100_iid_2000_skewed_gamma_${gamma}" \
            --local_ep 4 \
            --participation 0 \
            --logfile "$logfile" \
            --metrics_dir "$metrics_dir" \
            --gamma "$gamma"
    else
        python3 "$python_script" \
            --dataset cifar \
            --iid 1 \
            --epochs 2000 \
            --data_dir "$data_dir" \
            $gpu_arg \
            --local_ep 4 \
            --participation 0 \
            --logfile "$logfile" \
            --metrics_dir "$metrics_dir" \
            --gamma "$gamma"
    fi
done