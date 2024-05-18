#!/bin/bash

# 使用方式: ./run_model.sh /path/to/model.pth cub

MODEL_PATH='output_cosine/CD_CUB2011U/RankStat&WTA/checkpoints/epoch-best.pth'
DATASET_NAME='CD_CUB2011U'

python rs_results.py \
 --model_path $MODEL_PATH\
 --dataset $DATASET_NAME
