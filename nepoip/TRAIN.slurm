#!/bin/bash
#SBATCH -p mei3
#SBATCH -J example-train
#SBATCH -o SLURM/example-train.out
#SBATCH -e SLURM/example-train.err
#SBATCH --mem=18G
#SBATCH -t 3-0
#SBATCH -n 12

conda activate nepoip-test

nequip-train configs/ala_water_wB97x_decomposed_git.yaml
# python nequip/scripts/train.py configs/ala_water_wB97x_git.yaml