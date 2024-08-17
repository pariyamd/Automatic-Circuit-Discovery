#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=launch_acdc_%A_%a.out
#SBATCH --error=launch_acdc_%A_%a.err
#SBATCH --array=0-400

# Your commands go here
# module load anaconda/3
# conda deactivate
# # conda activate conda_jailbreak
# export PYTHONPATH="${PYTHONPATH}:/home/mila/p/paria.mehrbod/scratch/jailbreak_llm/Automatic-Circuit-Discovery"
# echo $PYTHONPATH

source setup.sh

command=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" acdc_commands_short.txt)
echo $command
eval $command