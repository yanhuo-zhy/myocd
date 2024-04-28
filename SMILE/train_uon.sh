#!/bin/bash 
#SBATCH --account cvl
#SBATCH -p amp20
#SBATCH --qos amp20
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/pszzz/hyzheng/myocd/temp/smile_Animalia_test.txt

module load gcc/gcc-10.2.0
# module load nvidia/cuda-10.0 nvidia/cudnn-v7.6.5.32-forcuda10.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/pszzz/miniconda3/bin/activate zhy
CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=./outputs/Animalia/

mkdir -p ${SAVE_DIR}

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m train_smile \
            --dataset_name 'Animalia' \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 100 \
            --base_model vit_dino \
            --num_workers 16 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.5 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.01 \
            --eval_funcs 'v1' 'v2' \
> ${SAVE_DIR}logfile_${EXP_NUM}.out