#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=launch_docstring_%A_%a.out
#SBATCH --error=launch_docstring_%A_%a.err
#SBATCH --array=0-167

# Your commands go here
# module load anaconda/3
# conda deactivate
# # conda activate conda_jailbreak
# export PYTHONPATH="${PYTHONPATH}:/home/mila/p/paria.mehrbod/scratch/jailbreak_llm/Automatic-Circuit-Discovery"
# echo $PYTHONPATH

source setup.sh

command=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" docstring_commands.txt)
echo $command
eval $command