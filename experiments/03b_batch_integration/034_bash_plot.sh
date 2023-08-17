#!/bin/bash
#SBATCH --job-name=dgd_b

#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1

#number of cpus we want to allocate for each program with memory used (per CPU or in total?)
#SBATCH --cpus-per-task=1 --mem=72000M

#requesting GPUs: name (titanx) and number (1)
#the current GPU options are: A100 (14 avail, 40 GB), A40 (10 Avail, 40 GB), titanrtx (48 avail, 24 GB), titanx (24 avail, 12 GB), testlak40, testlak20, gtx1080
#SBATCH -p gpu --gres=gpu:titanrtx:1

#Note that a program will be killed once it exceeds this time!
#SBATCH --time=50:00:00

#Skipping many options! see man sbatch
# From here on, we can start our program

# for GPU usage:
hostname
echo $CUDA_VISIBLE_DEVICES
###
# feel free to split these jobs up
###
# DGD
python ./figures/fig3_v2.py
python ./figures/supplementaries/fig_batchmodel.py
python ./figures/supplementaries/fig3_vis_supp.py