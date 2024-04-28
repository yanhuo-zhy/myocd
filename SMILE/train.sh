export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
SAVE_DIR=/wang_hp/zhy/OCD/source_code/outputs/food/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

python -m train_smile \
            --dataset_name 'food' \
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