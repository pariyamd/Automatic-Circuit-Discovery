#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --output=docstring_job_attn-only-4l.out
#SBATCH --error=docstring_job_attn-only-4l.err

# Your commands go here
module load anaconda/3
conda deactivate
conda activate conda_jailbreak
export PYTHONPATH="${PYTHONPATH}:/home/mila/p/paria.mehrbod/scratch/jailbreak_llm/Automatic-Circuit-Discovery"
echo $PYTHONPATH


python acdc/main.py --task=docstring --threshold=0.09500 --using-wandb --wandb-run-name=launch_docstring-acdc-000 --wandb-group-name=acdc-docstring --device=cuda --reset-network=0 --seed=516626229 --metric=kl_div --torch-num-threads=4 --wandb-dir=wandb_runs/ --wandb-mode=online --max-num-epochs=100000