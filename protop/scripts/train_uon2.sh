#!/bin/bash 
#SBATCH --account cvl
#SBATCH -p amp48
#SBATCH --qos amp48
#SBATCH -N 1
#SBATCH -c 5
#SBATCH --mem=20000
#SBATCH --gres gpu:1
#SBATCH -o /home/psawl/hyzheng/myocd/temp/my_protop_ablation_cub_nocenterloss.txt

module load gcc/gcc-10.2.0
module load nvidia/cuda-11.1 nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source /home/psawl/miniconda3/bin/activate zhy

CUDA_VISIBLE_DEVICES=0 python main.py \
    --base_architecture="deit_base_patch16_224" \
    --data_set="CD_CUB2011U" \
    --data_path="/db/psawl/cub" \
    --input_size=224 \
    --output_dir="output_cosine/CD_CUB2011U/ablation_nocenterloss_seed(1027)" \
    --batch_size=128 \
    --seed=1027 \
    --opt="adamw" \
    --sched="cosine" \
    --warmup-epochs=5 \
    --warmup-lr=1e-4 \
    --decay-epochs=10 \
    --decay-rate=0.1 \
    --weight_decay=0.05 \
    --epochs=200 \
    --finetune="protopformer" \
    --features_lr=1e-4 \
    --add_on_layers_lr=1e-3 \
    --prototype_vectors_lr=1e-3 \
    --prototype_shape 500 768 1 1 \
    --reserve_layers 11 \
    --reserve_token_nums 196 \
    --use_global=True \
    --use_ppc_loss=False \
    --ppc_cov_thresh=1. \
    --ppc_mean_thresh=2. \
    --global_coe=0.5 \
    --global_proto_per_class=5 \
    --ppc_cov_coe=0.1 \
    --ppc_mean_coe=0.5


CUDA_VISIBLE_DEVICES=0 python main.py \
    --base_architecture="deit_base_patch16_224" \
    --data_set="CD_CUB2011U" \
    --data_path="/db/psawl/cub" \
    --input_size=224 \
    --output_dir="output_cosine/CD_CUB2011U/ablation_nocenterloss_seed(1028)" \
    --batch_size=128 \
    --seed=1028 \
    --opt="adamw" \
    --sched="cosine" \
    --warmup-epochs=5 \
    --warmup-lr=1e-4 \
    --decay-epochs=10 \
    --decay-rate=0.1 \
    --weight_decay=0.05 \
    --epochs=200 \
    --finetune="protopformer" \
    --features_lr=1e-4 \
    --add_on_layers_lr=1e-3 \
    --prototype_vectors_lr=1e-3 \
    --prototype_shape 500 768 1 1 \
    --reserve_layers 11 \
    --reserve_token_nums 196 \
    --use_global=True \
    --use_ppc_loss=False \
    --ppc_cov_thresh=1. \
    --ppc_mean_thresh=2. \
    --global_coe=0.5 \
    --global_proto_per_class=5 \
    --ppc_cov_coe=0.1 \
    --ppc_mean_coe=0.5

CUDA_VISIBLE_DEVICES=0 python main.py \
    --base_architecture="deit_base_patch16_224" \
    --data_set="CD_CUB2011U" \
    --data_path="/db/psawl/cub" \
    --input_size=224 \
    --output_dir="output_cosine/CD_CUB2011U/ablation_nocenterloss_seed(1029)" \
    --batch_size=128 \
    --seed=1029 \
    --opt="adamw" \
    --sched="cosine" \
    --warmup-epochs=5 \
    --warmup-lr=1e-4 \
    --decay-epochs=10 \
    --decay-rate=0.1 \
    --weight_decay=0.05 \
    --epochs=200 \
    --finetune="protopformer" \
    --features_lr=1e-4 \
    --add_on_layers_lr=1e-3 \
    --prototype_vectors_lr=1e-3 \
    --prototype_shape 500 768 1 1 \
    --reserve_layers 11 \
    --reserve_token_nums 196 \
    --use_global=True \
    --use_ppc_loss=False \
    --ppc_cov_thresh=1. \
    --ppc_mean_thresh=2. \
    --global_coe=0.5 \
    --global_proto_per_class=5 \
    --ppc_cov_coe=0.1 \
    --ppc_mean_coe=0.5