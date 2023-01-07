#!/usr/bin/env python3

import os
import torch

import sys
sys.path.append('/home/lanbo/wifi_wavelet')
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from scripts.utils import *

# tmux new -s wifi
# tmux a -t wifi
# /home/lanbo/anaconda3/envs/test/bin/python3 -u /home/lanbo/wifi_wavelet/scripts/vit_run.py

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda = 3

    # os.system('tmux a -t wifi_vio')

    config = DatasetDefaultConfig()

    model_list = [
        # ('vit_b_2', 'vit_span_cls_freq', 128),
        # ('vit_b_4', 'vit_span_cls_freq', 128),
        # ('vit_b_8', 'vit_span_cls_freq', 128),

        # ('vit_b_16', 'vit_span_cls_raw', 128),
        # ('vit_b_32', 'vit_span_cls_raw', 128),
        # ('vit_b_64', 'vit_span_cls_raw', 128),
        #
        # ('vit_s_2', 'vit_span_cls_freq', 128),
        # ('vit_s_4', 'vit_span_cls_freq', 128),
        # ('vit_s_8', 'vit_span_cls_freq', 128),

        # ('vit_s_16_0.5', 'vit_span_cls_raw', 64),
        # ('vit_s_32', 'vit_span_cls_raw', 64),


        # ('vit_b_16_0.5', 'vit_span_cls_raw', 64),
        ('vit_b_16_0.6', 'vit_span_cls_raw', 128),
        ('vit_b_16_0.7', 'vit_span_cls_raw', 128),
        ('vit_b_16_0.8', 'vit_span_cls_raw', 128),
        ('vit_b_16_0.9', 'vit_span_cls_raw', 128),

        # ('vit_b_32_0.5', 'vit_span_cls_raw', 128),
        # ('vit_b_64', 'vit_span_cls_raw', 128),

        # ('vit_s_64', 'vit_span_cls_raw', 64),
        #
        # ('vit_ms_2', 'vit_span_cls_freq', 128),
        # ('vit_ms_4', 'vit_span_cls_freq', 128),
        # ('vit_ms_8', 'vit_span_cls_freq', 128),
        # ('vit_ms_16_0.6', 'vit_span_cls_raw', 64),
        # ('vit_ms_32_0.6', 'vit_span_cls_raw', 64),
        # ('vit_ms_64', 'vit_span_cls_raw', 128),
        #
        # ('vit_es_2', 'vit_span_cls_freq', 128),
        # ('vit_es_4', 'vit_span_cls_freq', 128),
        # ('vit_es_8', 'vit_span_cls_freq', 128),
        # ('vit_es_16_0.6', 'vit_span_cls_raw', 64),
        # ('vit_es_32_0.6', 'vit_span_cls_raw', 64),
        # ('vit_es_64', 'vit_span_cls_raw', 128),
    ]
    config.dataset_list.append(f'WiVio')
    print(config.dataset_list)
    for dataset_name in config.dataset_list:
        for module in model_list:
            """
                修改log输出地址，datasource_path 在 sh 里面改
            """
            log_name = 'log_dataset3'
            tab = 'dataset3'
            datasource_path = '/home/lanbo/dataset/wifi_violence_processed3/'

            log_path = os.path.join('/home/lanbo/wifi_wavelet/log', log_name)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]

            eval_batch_size = 1
            num_epoch = 2

            opt_method = "adamw"
            lr_rate = 2e-4
            weight_decay = 1e-4
            lr_rate_adjust_epoch = 100
            lr_rate_adjust_factor = 0.2
            save_epoch = 501
            eval_epoch = 501
            patience = 50

            test_batch_size = batch_size
            train_batch_size = batch_size
            # print(
            #     './script_run.sh %d %s %s %s %s %d' %
            #     (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size)
            # )

            # os.chdir("/home/lanbo/wifi_wavelet/")
            # os.system('conda activate test')
            # os.system(
            #     f'python main.py --dataset_name "{dataset_name}" --gpu_device {cuda} \
            #     --backbone_name "{backbone_name}" --head_name "{head_name}" --strategy_name "{strategy_name}" \
            #     --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --num_epoch {num_epoch} \
            #     --opt_method {opt_method} --lr_rate {lr_rate} --weight_decay {weight_decay} --datasource_path {datasource_path}\
            #     --lr_rate_adjust_epoch {lr_rate_adjust_epoch} --lr_rate_adjust_factor {lr_rate_adjust_factor}  \
            #     --save_epoch {save_epoch} --eval_epoch {eval_epoch} --patience {patience} --is_train true \
            #     > {dataset_name}-{backbone_name}-{strategy_name}-TRAIN.log'
            # )

            # os.system(f'PYTHON main.py --dataset_name "{dataset_name}" --gpu_device {cuda} \
            #             --backbone_name "{backbone_name}" --head_name "{head_name}" --strategy_name "{strategy_name}" \
            #             --test_batch_size {test_batch_size} \
            #             > {dataset_name}-{strategy_name}-TEST.log')



            os.system(
                'bash /home/lanbo/wifi_wavelet/scripts/script_run.sh %d %s %s %s %s %d %s %s %s' %
                (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size, log_path, datasource_path, tab)
            )