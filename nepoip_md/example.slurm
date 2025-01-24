#!/bin/bash
#SBATCH -p mei3
#SBATCH -J nepoip_md_1
#SBATCH -o example_run/nepoip_md.out
#SBATCH -e example_run/nepoip_md.err
#SBATCH --mem=12G
#SBATCH -t 7-0
#SBATCH -n 8

conda activate nepoip-test

python example_run.py --traj 1
