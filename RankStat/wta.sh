#!/bin/bash 
#SBATCH --account cvl
#SBATCH -p general
#SBATCH --qos normal
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/myocd/temp/wta_Mollusca.txt

module load gcc/gcc-10.2.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/pszzz/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0

MODEL_PATH='output_cosine/Mollusca/WTA/checkpoints/epoch-best.pth'
DATASET_NAME='Mollusca'

python wta.py \
 --model_path $MODEL_PATH\
 --dataset $DATASET_NAME
