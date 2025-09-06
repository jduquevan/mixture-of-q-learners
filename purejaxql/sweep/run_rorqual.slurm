#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=ppomix

# Remove one # to uncommment
#SBATCH --output=/project/rrg-bengioy-ad/jduque/slurm_output/slurm-%j.out
#SBATCH --error=/project/rrg-bengioy-ad/jduque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=0-11:59:00   
#SBATCH --gpus=h100:1

# Submit jobs.
cd /project/rrg-bengioy-ad/jduque/mixture-of-q-learners
module purge
module load cuda/12.2
export HOME=/project/rrg-bengioy-ad/jduque/
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR=/project/rrg-bengioy-ad/jduque/wandb
source /project/rrg-bengioy-ad/jduque/mixture-of-q-learners/.venv/bin/activate

# Define all arguments
SEED=${1}
LR=${2}
HIDDEN_SIZE=${3}
NUM_STEPS=${4}
EPS_FINISH=${5}

# Construct the full command
CMD="CUDA_VISIBLE_DEVICES=0 python  /project/rrg-bengioy-ad/jduque/mixture-of-q-learners/purejaxql/pqn_atari_lstm.py \
    SEED=${SEED} \
    alg.LR=${LR} \
    alg.HIDDEN_SIZE=${HIDDEN_SIZE} \
    alg.NUM_STEPS=${NUM_STEPS} \
    alg.EPS_FINISH=${EPS_FINISH} \
    alg.NUM_ENVS=512 \
    alg.ENV_NAME=Skiing-v5"

# Print the full command for debugging
stdbuf -o0 echo "Executing command: $CMD"

# Execute the command
eval $CMD