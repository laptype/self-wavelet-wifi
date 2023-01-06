#!/bin/bash

# ./script_run [cuda] [dataset_name] [backbone_name] [head_name] [strategy_name] [batch_size]

python="/home/lanbo/anaconda3/envs/test/bin/python3"

cuda=$1
dataset_name=$2

backbone_name=$3
head_name=$4
strategy_name=$5

train_batch_size=$6
eval_batch_size=1
num_epoch=400

opt_method="adamw"
lr_rate=2e-4
weight_decay=1e-4
lr_rate_adjust_epoch=100
lr_rate_adjust_factor=0.2
save_epoch=501
eval_epoch=501
patience=50

test_batch_size=$6

log_path=$7

datasource_path=$8

#echo "========================${dataset_name}-${backbone_name}-${strategy_name}-TRAIN========================"
#CUDA_VISIBLE_DEVICES=${cuda} ${python} /home/lanbo/wifi_wavelet/main.py --dataset_name ${dataset_name} --gpu_device ${cuda} \
#--backbone_name ${backbone_name} --head_name ${head_name} --strategy_name ${strategy_name} \
#--train_batch_size ${train_batch_size} --eval_batch_size ${eval_batch_size} --num_epoch ${num_epoch} \
#--opt_method ${opt_method} --lr_rate ${lr_rate} --weight_decay ${weight_decay} \
#--lr_rate_adjust_epoch ${lr_rate_adjust_epoch} --lr_rate_adjust_factor ${lr_rate_adjust_factor}  \
#--save_epoch ${save_epoch} --eval_epoch ${eval_epoch} --patience ${patience} --is_train true \
#--datasource_path "/home/lanbo/dataset/wifi_violence_processed/"\
#--tab "tab1"\
#> ${log_path}/${dataset_name}-${backbone_name}-${strategy_name}-TRAIN.log

echo "========================${dataset_name}-${strategy_name}-TEST========================"
CUDA_VISIBLE_DEVICES=${cuda} ${python} ../main.py --dataset_name ${dataset_name} --gpu_device ${cuda} \
--backbone_name ${backbone_name} --head_name ${head_name} --strategy_name ${strategy_name} \
--test_batch_size ${test_batch_size} \
--datasource_path "${datasource_path}"\
> ${log_path}/${dataset_name}-${backbone_name}-${strategy_name}-TEST.log