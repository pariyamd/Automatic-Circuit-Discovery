#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=5:00:00
#SBATCH --output=induction_job_%j.out
#SBATCH --error=induction_job_%j.err
#SBATCH --job-name=induction

# Your commands go here
module load anaconda/3
conda deactivate
conda activate conda_jailbreak
export PYTHONPATH="${PYTHONPATH}:/home/mila/p/paria.mehrbod/scratch/jailbreak_llm/Automatic-Circuit-Discovery"
echo $PYTHONPATH


python acdc/main.py --task=induction --threshold=0.56230 --using-wandb --wandb-run-name=launch_induction-acdc-000 --wandb-group-name=acdc-induction --device=cuda --reset-network=0 --seed=424671755 --metric=kl_div --torch-num-threads=4 --wandb-dir=wandb_runs/ --wandb-mode=online

python acdc/main.py --task=induction --threshold=0.56230 --using-wandb --wandb-run-name=launch_induction-acdc-001 --wandb-group-name=acdc-induction --device=cuda --reset-network=0 --seed=424671755 --metric=kl_div --torch-num-threads=4 --wandb-dir=wandb_runs/ --wandb-mode=online --zero-ablation
python acdc/main.py --task=induction --threshold=0.56230 --using-wandb --wandb-run-name=launch_induction-acdc-002 --wandb-group-name=acdc-induction --device=cuda --reset-network=1 --seed=424671755 --metric=kl_div --torch-num-threads=4 --wandb-dir=wandb_runs/ --wandb-mode=online
python acdc/main.py --task=induction --threshold=0.56230 --using-wandb --wandb-run-name=launch_induction-acdc-003 --wandb-group-name=acdc-induction --device=cuda --reset-network=1 --seed=424671755 --metric=kl_div --torch-num-threads=4 --wandb-dir=wandb_runs/ --wandb-mode=online --zero-ablation