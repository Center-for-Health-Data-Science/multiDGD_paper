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
#SBATCH --time=02:00:00

#Skipping many options! see man sbatch
# From here on, we can start our program

# for GPU usage:
hostname
echo $CUDA_VISIBLE_DEVICES
###
# feel free to split these jobs up
###

# testing without correction model error
# first over batches, then over random seeds

#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 0 --random_seed 0
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 1 --random_seed 0
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 2 --random_seed 0
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 3 --random_seed 0

#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 0 --random_seed 37
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 1 --random_seed 37
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 2 --random_seed 37
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 3 --random_seed 37

#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 0 --random_seed 8790
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 1 --random_seed 8790
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 2 --random_seed 8790
#python ./testing/revision_human_bonemarrow_freeCovariateModel.py --batch_left_out 3 --random_seed 8790

# testing without correction model error and from zero
# first over batches, then over random seeds

python ./testing/revision_human_bonemarrow_freeCovariateModel_fromZero.py --batch_left_out 0 --random_seed 0
#python ./testing/revision_human_bonemarrow_freeCovariateModel_fromZero.py --batch_left_out 1 --random_seed 0
#python ./testing/revision_human_bonemarrow_freeCovariateModel_fromZero.py --batch_left_out 2 --random_seed 0
#python ./testing/revision_human_bonemarrow_freeCovariateModel_fromZero.py --batch_left_out 3 --random_seed 0
