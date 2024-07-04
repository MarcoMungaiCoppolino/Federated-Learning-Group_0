#!/bin/bash

# Check if GPU is available
if command -v nvidia-smi &> /dev/null
then
    GPU=0
else
    GPU=-1
fi

# Set wandb parameters if available
WANDB_USERNAME=${1:-}
WANDB_KEY=${2:-}

# Loop over the combinations of j and nc
for j in 4 8 16
do
    for nc in 1 5 10 50
    do
        name="federated_cifar_100_noniid_j=${j}_nc=${nc}"
        
        # Base command
        CMD="python3 Federated-Learning-Group_0/src/federated.py \
            --dataset cifar \
            --epochs 2000 \
            --data_dir /content/drive/MyDrive/MLDL/cifar/data \
            --local_ep $j \
            --Nc $nc \
            --participation 1 \
            --logfile $name"
        
        # Add GPU parameter if GPU is available
        if [ "$GPU" -ne -1 ]; then
            CMD="$CMD --gpu $GPU"
        fi

        # Add wandb parameters if they are provided
        if [ -n "$WANDB_USERNAME" ] && [ -n "$WANDB_KEY" ]; then
            CMD="$CMD --wandb_key $WANDB_KEY \
                --wandb_username $WANDB_USERNAME \
                --wandb_project Federated_Learning \
                --wandb_run_name $name"
        fi

        # Execute the command
        echo "Executing: $CMD"
        eval $CMD
    done
done