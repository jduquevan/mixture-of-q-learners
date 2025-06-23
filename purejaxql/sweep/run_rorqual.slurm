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
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-23:59:00   
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1

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
PPO_EPOCHS=${2}
LR=${3}
ENT=${4}

# Construct the full command
CMD="CUDA_VISIBLE_DEVICES=0 python  /project/rrg-bengioy-ad/jduque/mixture-of-q-learners/purejaxql/ppomix.py \
    SEED=${SEED} \
    alg.NUM_EPOCHS=${PPO_EPOCHS} \
    alg.LR=${LR} \
    alg.ENT_COEFF=${ENT} \
    alg.NUM_ENVS=16 \
    alg.NUM_AGENTS=4 \
    alg.ENV_NAME=Breakout-v5"

# Print the full command for debugging
stdbuf -o0 echo "Executing command: $CMD"

# Execute the command
eval $CMD