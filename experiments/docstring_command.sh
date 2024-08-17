#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=docstring_seccache.out
#SBATCH --error=docstring_seccache.err

# Your commands go here
module load anaconda/3
conda deactivate
conda activate conda_jailbreak
export PYTHONPATH="${PYTHONPATH}:/home/mila/p/paria.mehrbod/scratch/jailbreak_llm/Automatic-Circuit-Discovery"
echo $PYTHONPATH


python acdc/main.py --task=docstring --threshold=0.09500 --wandb-run-name=launch_docstring-acdc-seccache --wandb-group-name=acdc-docstring-test --device=cuda --reset-network=0 --seed=516626229 --metric=kl_div --torch-num-threads=4 --wandb-dir=wandb_runs/ --wandb-mode=online --max-num-epochs=10000  --first-cache-cpu=False --second-cache-cpu=True