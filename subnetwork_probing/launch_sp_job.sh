#!/bin/bash
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=launch_sp_%A_%a.out
#SBATCH --error=launch_sp_%A_%a.err
#SBATCH --array=0-130

# Your commands go here
# module load anaconda/3
# conda deactivate
# # conda activate conda_jailbreak
# export PYTHONPATH="${PYTHONPATH}:/home/mila/p/paria.mehrbod/scratch/jailbreak_llm/Automatic-Circuit-Discovery"
# echo $PYTHONPATH

source setup.sh

command=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" sp_commands_short.txt)
echo $command
eval $command