#!/bin/bash
#SBATCH -p mei4
#SBATCH -J nepoip_md_1
#SBATCH -o nepoip_md.out
#SBATCH -e nepoip_md.err
#SBATCH --mem=12G
#SBATCH -t 7-0
#SBATCH -n 8

conda activate nepoip-test

python example_run_cut.py --traj 1
