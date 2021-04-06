#!/bin/bash
#SBATCH --account=def-hamarneh 
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=0-16:00

source ../tensorflow1/bin/activate
cd /home/saurabh/Deep-structured-facial-landmark-detection/ 

python -i eval_fan_crf_single_final.py filedir/coord_1200_file${1}.txt 
