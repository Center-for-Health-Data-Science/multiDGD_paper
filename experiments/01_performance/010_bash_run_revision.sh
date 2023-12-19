#!/bin/bash
#SBATCH --job-name=oDGD

#number of independent tasks we are going to start in this script
#SBATCH --ntasks=1

#number of cpus we want to allocate for each program with memory used (per CPU or in total?)
#SBATCH --cpus-per-task=1 --mem=48000M

#requesting GPUs: name (titanx) and number (1)
#the current GPU options are: A100 (14 avail, 40 GB), A40 (10 Avail, 40 GB), titanrtx (48 avail, 24 GB), titanx (24 avail, 12 GB), testlak40, testlak20, gtx1080
#SBATCH -p gpu --gres=gpu:titanrtx:1

#Note that a program will be killed once it exceeds this time!
#SBATCH --time=01:00:00

#Skipping many options! see man sbatch
# From here on, we can start our program

# for GPU usage:
hostname
echo $CUDA_VISIBLE_DEVICES
###
# feel free to split these jobs up
###
# run each model training with 3 different random seeds
# DGD
#python3 ./training/revision_mouse_gastrulation_k1.py --random_seed 0 --n_components 1
#python3 ./training/revision_mouse_gastrulation_k1.py --random_seed 37 --n_components 1
#python3 ./training/revision_mouse_gastrulation_k1.py --random_seed 8790 --n_components 1

#python3 ./training/revision_human_bonemarrow_k1.py --random_seed 0 --n_components 1
#python3 ./training/revision_human_bonemarrow_k1.py --random_seed 37 --n_components 1
#python3 ./training/revision_human_bonemarrow_k1.py --random_seed 8790 --n_components 1

#python3 ./training/revision_human_brain_k1.py --random_seed 0 --n_components 1
#python3 ./training/revision_human_brain_k1.py --random_seed 37 --n_components 1
#python3 ./training/revision_human_brain_k1.py --random_seed 8790 --n_components 1

###

# run now without covariate to be able to compare to scDGD

#python3 ./training/revision_human_bonemarrow_noCovariate.py --random_seed 0
#python3 ./training/revision_human_bonemarrow_noCovariate.py --random_seed 37
#python3 ./training/revision_human_bonemarrow_noCovariate.py --random_seed 8790

#python3 ./training/revision_mouse_gastrulation_noCovariate.py --random_seed 0
#python3 ./training/revision_mouse_gastrulation_noCovariate.py --random_seed 37
#python3 ./training/revision_mouse_gastrulation_noCovariate.py --random_seed 8790

###

# run with RNA only

#python3 ./training/revision_human_bonemarrow_RNAonly.py --random_seed 0
#python3 ./training/revision_human_bonemarrow_RNAonly.py --random_seed 37
#python3 ./training/revision_human_bonemarrow_RNAonly.py --random_seed 8790

#python3 ./training/revision_mouse_gastrulation_RNAonly.py --random_seed 0
#python3 ./training/revision_mouse_gastrulation_RNAonly.py --random_seed 37
#python3 ./training/revision_mouse_gastrulation_RNAonly.py --random_seed 8790

#python3 ./training/revision_human_brain_RNAonly.py --random_seed 0
#python3 ./training/revision_human_brain_RNAonly.py --random_seed 37
#python3 ./training/revision_human_brain_RNAonly.py --random_seed 8790

###

# run with RNA only and no covariate (scDGD)

#python3 ./training/revision_human_bonemarrow_scDGD.py --random_seed 0
#python3 ./training/revision_human_bonemarrow_scDGD.py --random_seed 37
#python3 ./training/revision_human_bonemarrow_scDGD.py --random_seed 8790

#python3 ./training/revision_mouse_gastrulation_scDGD.py --random_seed 0
#python3 ./training/revision_mouse_gastrulation_scDGD.py --random_seed 37
#python3 ./training/revision_mouse_gastrulation_scDGD.py --random_seed 8790

#python3 ./training/revision_human_brain_scDGD.py --random_seed 0
#python3 ./training/revision_human_brain_scDGD.py --random_seed 37
#python3 ./training/revision_human_brain_scDGD.py --random_seed 8790

###

# analyis
python ./analysis/revision_human_bonemarrow_predictions.py