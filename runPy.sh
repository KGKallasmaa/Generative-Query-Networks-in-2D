#!/bin/bash
#SBATCH -J gen-net-job
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvinmeltsov@gmail.com
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:2
#SBATCH --mem=128000
#SBATCH --cpus-per-task=6

module load python/3.6.3/CUDA-9.0

# This is a runner script for running some python scripts in SLURM
# Use for bigger jobs.
#
# TODO: Activate the gqn enviroment first!
# TODO: EMAIL and TIME fields!

# CUDA for GPUs
export CUDA_VISIBLE_DEVICES=0,1

# start TensorBoard in background for logging
tensorboard --logdir "./log" &
TENSORBOARD_PID=$!
echo "Started Tensorboard with PID: $TENSORBOARD_PID"

# Start training script
python run-gqn.py \
    --data_dir "./shepard_metzler_5_parts" \
    --n_epochs 20 \
    --fraction 0.1 \
    --data_parallel "True" \
    --workers 6 \
    --log_dir "./log"
