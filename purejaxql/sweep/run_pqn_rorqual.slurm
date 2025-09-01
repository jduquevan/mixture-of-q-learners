#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=pqn

# Remove one # to uncommment
#SBATCH --output=/project/rrg-bengioy-ad/jduque/slurm_output/slurm-%j.out
#SBATCH --error=/project/rrg-bengioy-ad/jduque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=0-15:59:00   
#SBATCH --gpus=h100:1

# Submit jobs.
cd /project/rrg-bengioy-ad/jduque/mixture-of-q-learners
module purge
module load cuda/12.2
export HOME=/project/rrg-bengioy-ad/jduque/
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR=/project/rrg-bengioy-ad/jduque/wandb
export WANDB_TAGS="Scaling_experiment"
source /project/rrg-bengioy-ad/jduque/mixture-of-q-learners/.venv/bin/activate

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch sweep/run_rorqual.slurm -- python /path/ppomix_se.py KEY=VAL ..."
  exit 1
fi

echo "Executing command:"
printf ' %q' "$@"; echo
exec "$@"