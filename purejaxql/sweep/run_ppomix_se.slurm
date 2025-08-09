#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=ppo_mix

# Remove one # to uncomment
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

# Define how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --mem=40G
#SBATCH --time=0-23:59:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=16

# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda deactivate
module load cuda/12.0
export HYDRA_FULL_ERROR=1
source ../.venv/bin/activate

# Define all arguments
SEED=${1}
PPO_EPOCHS=${2}
LR=${3}
ENT=${4}

# Construct the full command
CMD="python ppomix.py \
    SEED=${SEED} \
    alg.NUM_EPOCHS=${PPO_EPOCHS} \
    alg.LR=${LR} \
    alg.ENT_COEFF=${ENT} \
    alg.NUM_ENVS=512 \
    alg.ENV_NAME=Gravitar-v5"

# Print the full command for debugging
stdbuf -o0 echo "Executing command: $CMD"

# Execute the command
eval $CMD