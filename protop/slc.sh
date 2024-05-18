#!/bin/bash 
#SBATCH --account cvl
#SBATCH -p general
#SBATCH --qos normal
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/myocd/temp/slc_Arachnida.txt

module load gcc/gcc-10.2.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/pszzz/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0

MODEL_PATH='output_cosine/Arachnida/comparison_slc_seed(1027)/checkpoints/epoch-best.pth'
DATASET_NAME='Arachnida'

python slc_results.py \
 --model_path $MODEL_PATH\
 --dataset $DATASET_NAME
