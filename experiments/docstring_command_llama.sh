#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=docstring_job_sp.out
#SBATCH --error=docstring_job_sp.err
#SBATCH --partition=short-unkillable

# Your commands go here
module load anaconda/3
conda deactivate
conda activate conda_jailbreak
export PYTHONPATH="${PYTHONPATH}:/home/mila/p/paria.mehrbod/scratch/jailbreak_llm/Automatic-Circuit-Discovery"
echo $PYTHONPATH


# python acdc/main.py --task=docstring --threshold=0.09500 --using-wandb --wandb-run-name=launch_docstring-acdc-001 --wandb-group-name=acdc-docstring --device=cuda --reset-network=0 --seed=516626229 --metric=kl_div --torch-num-threads=4 --wandb-dir=wandb_runs/ --wandb-mode=online --max-num-epochs=100000 --model-type="llama" --zero-ablation --first-cache-cpu=False --second-cache-cpu=False


python subnetwork_probing/train.py --task=docstring --device=cuda --epochs=100000 --zero-ablation=1 --seed=3205774520 --loss-type=kl_div --num-examples=50 --seq-len=41 --n-loss-average-runs=20 --torch-num-threads=4 --wandb-name=launch_docstring-sp-004 --wandb-group=acdc-docstring --wandb-dir=wandb_runs --wandb-mode=online --reset-subject=0

# --reset-subject=0 
