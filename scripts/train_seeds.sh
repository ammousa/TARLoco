#!/bin/bash
# RUN BY CALLING: bash scripts/train_seeds.sh X
#                 nohup bash scripts/train_seeds.sh X > logs/nohup/train_seeds.log 2>&1 &
# Kill by calling: pkill -f train_seeds.sh
#                  ps -ef | grep train_seeds.sh | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Check if CUDA device is provided
if [ -z "$1" ]; then
    echo "Error: CUDA device not specified."
    echo "Usage: bash scripts/train_seeds.sh <cuda_device>"
    exit 1
fi

CUDA_DEVICE=$1

# Define the list of unique commands (excluding --seed as it will be replaced)
commands=(
    "python standalone/tarloco/train.py --task go1-train-tar-rnn-rough --max_iterations 20000 --headless --logger wandb --group 'OURS'"

)

# Define the seeds to use
seeds=(0 41 1125)

# Summary of execution
echo "========================"
echo "Running on CUDA device: $CUDA_DEVICE"
echo "Number of commands: ${#commands[@]}"
echo "Seeds: ${seeds[@]}"
echo "========================"

# Iterate over each command and execute for each seed
for cmd in "${commands[@]}"; do
    for seed in "${seeds[@]}"; do
        # Extract the group value from the command
        group=$(echo "$cmd" | grep -oP "(?<=--group ')[^']+")
        # Construct the note value with the group and the dynamic seed
        note="$group S$seed"
        # Construct the full command
        full_cmd="$cmd --device cuda:$CUDA_DEVICE --note '$note' --seed $seed"
        echo "---------------------------"
        echo "Executing on CUDA device $CUDA_DEVICE: $full_cmd"
        echo "---------------------------"
        eval $full_cmd
        printf '\n%.0s' {1..5}
    done
done

echo "All commands completed on CUDA device $CUDA_DEVICE."
