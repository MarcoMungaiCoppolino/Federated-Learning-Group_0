#!/bin/bash

wandb_key="$1"  # First argument is the Wandb key
wandb_username="$2"  # Second argument is the Wandb username

# Set other variables as needed
python_script="/content/Federated-Learning-Group_0/src/federated.py"
data_dir="/content/drive/MyDrive/MLDL/shakespeare/data"
checkpoint_path="/content/drive/MyDrive/MLDL/shakespeare/checkpoints"
logfile_base="/content/drive/MyDrive/MLDL/shakespeare/logs/federated_shakespeare_100_noniid_skewed"
metrics_dir="/content/drive/MyDrive/MLDL/shakespeare/metrics"

# Check if GPU is available (using nvidia-smi command to check)
if nvidia-smi &> /dev/null; then
    gpu_arg="--gpu 0"
else
    gpu_arg=""
fi

# Array of gamma values
gammas=(0.1 0.5 0.7)

# Loop over the combinations of j, nc, and gamma
for j in 4 8 16
do
    for gamma in "${gammas[@]}"
    do
        name="federated_shakespeare_100_noniid_skewed_j=${j}_nc=${nc}_gamma=${gamma}"
        logfile="${logfile_base}_j=${j}_nc=${nc}_gamma=${gamma}.log"
        
        # Base command
        CMD="python3 $python_script \
            --dataset shakespeare \
            --model lstm \
            --checkpoint_path $checkpoint_path \
            --epochs 2000 \
            --data_dir $data_dir \
            --local_ep $j \
            --lr 1 \
            --participation 1 \
            --logfile $logfile \
            --metrics_dir $metrics_dir \
            --gamma $gamma"
        
        # Add GPU parameter if GPU is available
        if [ -n "$gpu_arg" ]; then
            CMD="$CMD $gpu_arg"
        fi

        # Add wandb parameters if they are provided
        if [ -n "$wandb_key" ] && [ -n "$wandb_username" ]; then
            CMD="$CMD --wandb_key $wandb_key \
                --wandb_username $wandb_username \
                --wandb_project Federated_Learning \
                --wandb_run_name $name"
        fi

        # Execute the command
        echo "Executing: $CMD"
        eval $CMD
    done
done