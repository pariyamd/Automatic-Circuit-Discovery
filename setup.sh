#!/bin/bash

conda deactivate
conda activate conda_jailbreak
export PYTHONPATH="${PYTHONPATH}:/home/mila/p/paria.mehrbod/scratch/jailbreak_llm/Automatic-Circuit-Discovery"
echo $PYTHONPATH
