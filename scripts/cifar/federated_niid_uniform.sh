#!/bin/bash

wandb_key="$1"  # First argument is the Wandb key
wandb_username="$2"  # Second argument is the Wandb username

# Set other variables as needed
python_script="./src/federated.py"
data_dir="./data"
checkpoint_path="/Users/ace/Desktop/MLDL_resources/new_federated/checkpoints"
logfile_base="./logs/federated_cifar_100_noniid_uniform"
metrics_dir="./cifar/metrics"

# Check if GPU is available (using nvidia-smi command to check)
if nvidia-smi &> /dev/null; then
    gpu_arg="--gpu 0"
else
    gpu_arg=""
fi

# Loop over the combinations of j and nc
for j in 16 8 4
do
    for nc in 50 10 5 1
    do
        name="federated_cifar_100_noniid_uniform_j=${j}_nc=${nc}"
        logfile="${logfile_base}_j=${j}_nc=${nc}.log"
        
        # Base command
        CMD="/Users/ace/anaconda3/envs/fed/bin/python $python_script \
            --dataset cifar \
            --epochs 2000 \
            --checkpoint_path $checkpoint_path \
            --data_dir $data_dir \
            --local_ep $j \
            --Nc $nc \
            --participation 1 \
            --logfile $logfile \
            --metrics_dir $metrics_dir"
        
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