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

        # ('vit_b_16_0.2', 'vit_span_cls_raw', 64),
        # ('vit_b_16_0.3', 'vit_span_cls_raw', 64),
        # ('vit_b_16_0.4', 'vit_span_cls_raw', 64),
        # ('vit_b_16_0.5', 'vit_span_cls_raw', 64),
        # ('vit_b_16_0.6', 'vit_span_cls_raw', 64),
        # ('vit_b_16_0.7', 'vit_span_cls_raw', 64),

        # patch_size
        # ('vit_b_8_0.5', 'vit_span_cls_raw', 64),
        # ('vit_b_32_0.5', 'vit_span_cls_raw', 64),
        # ('vit_b_64_0.5', 'vit_span_cls_raw', 64),

        # ('resnet1d_101', 'resnet1d_span_cls_raw_time', 128),
        # ('resnet1d_50', 'resnet1d_span_cls_raw_time', 128),
        # ('resnet1d_34', 'resnet1d_span_cls_raw_time', 128),
        # ('resnet1d_18', 'resnet1d_span_cls_raw_time', 128),
        #
        # ('vit_s_16_0.4', 'vit_span_cls_raw', 64),
        # ('vit_s_16_0.5', 'vit_span_cls_raw', 64),
        # ('vit_s_32_0.5', 'vit_span_cls_raw', 64),

        # ('wavevit_wavelh2_0_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wavelh2_4_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wavelh2_8_s_16_0.4_0.1', 'vit_span_cls_raw', 64),

        # ('wavevit_wavelh_0_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wavelh_4_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wavelh_8_s_16_0.4_0.1', 'vit_span_cls_raw', 64),

        # ('wavevit_wave2_0_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wave2_4_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wave2_8_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        #
        # ('wavevit_wave2_4_s_32_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wave2_8_s_32_0.4_0.1', 'vit_span_cls_raw', 64),

        # ('wavevit_wavelh2_0_s_16_0.4_0.0', 'vit_span_cls_raw', 64),
        # ('wavevit_wavelh2_0_s_16_0.4_0.2', 'vit_span_cls_raw', 64),
        # ('wavevit_wavelh2_0_s_16_0.4_0.3', 'vit_span_cls_raw', 64),

        # ('wavevit_wave22_4_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wave22_4_s_16_0.5_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_wave22_4_s_16_0.6_0.1', 'vit_span_cls_raw', 64),
        #
        # ('wavevit_wave22_4_s_16_0.5_0.2', 'vit_span_cls_raw', 64),
        ('wavevit_wave22_4_s_16_0.5_0.3', 'vit_span_cls_raw', 64),

        # ('wavevit_test_4_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_test_8_s_16_0.4_0.1', 'vit_span_cls_raw', 64),
        # ('wavevit_test_0_s_16_0.4_0.1', 'vit_span_cls_raw', 64),

        # ('vit_b_32_0.4', 'vit_span_cls_raw', 64),

        # ('vit_ms_16_0.2', 'vit_span_cls_raw', 64),
        # ('vit_ms_16_0.5', 'vit_span_cls_raw', 64),
        # ('vit_ms_16_0.6', 'vit_span_cls_raw', 64),
        #
        # ('vit_es_16_0.2', 'vit_span_cls_raw', 64),
        # ('vit_es_16_0.5', 'vit_span_cls_raw', 64),
        # ('vit_es_16_0.6', 'vit_span_cls_raw', 64),

        # ('vit_b_32_0.5', 'vit_span_cls_raw', 128),
        # ('vit_b_64', 'vit_span_cls_raw', 128),

        # ('vit_s_64', 'vit_span_cls_raw', 64),
        #
        # ('vit_ms_2', 'vit_span_cls_freq', 128),
        # ('vit_ms_4', 'vit_span_cls_freq', 128),
        # ('vit_ms_8', 'vit_span_cls_freq', 128),
        # ('vit_ms_16_0.6', 'vit_span_cls_raw', 64),
        # ('vit_ms_16_0.7', 'vit_span_cls_raw', 64),
        # ('vit_ms_16_0.8', 'vit_span_cls_raw', 64),
        # ('vit_ms_64', 'vit_span_cls_raw', 128),
        #
        # ('vit_es_2', 'vit_span_cls_freq', 128),
        # ('vit_es_4', 'vit_span_cls_freq', 128),
        # ('vit_es_8', 'vit_span_cls_freq', 128),
        # ('vit_es_16_0.6', 'vit_span_cls_raw', 64),
        # ('vit_es_16_0.7', 'vit_span_cls_raw', 64),
        # ('vit_es_16_0.8', 'vit_span_cls_raw', 64),
        # ('vit_es_64', 'vit_span_cls_raw', 128),
    ]
    config.dataset_list.append(f'WiVio')
    print(config.dataset_list)
    for dataset_name in config.dataset_list:
        for module in model_list:

            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]

            log_name = 'day_2_3_wave'
            tab = 'day_2_3_wave'
            datasource_path = '/home/lanbo/dataset/wifi_violence_processed_loc/'

            log_path = os.path.join('/home/lanbo/wifi_wavelet/log', log_name)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            os.system(
                'bash /home/lanbo/wifi_wavelet/scripts/script_run.sh %d %s %s %s %s %d %s %s %s' %
                (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size, log_path, datasource_path, tab)
            )

            # ----------------------------------------------------
            # log_name = 'day_1_13_lin'
            # tab = 'day_1_13_lin'
            # datasource_path = '/home/lanbo/dataset/wifi_violence_processed_loc_lin/'
            #
            # log_path = os.path.join('/home/lanbo/wifi_wavelet/log', log_name)
            # if not os.path.exists(log_path):
            #     os.makedirs(log_path)
            #
            # os.system(
            #     'bash /home/lanbo/wifi_wavelet/scripts/script_run.sh %d %s %s %s %s %d %s %s %s' %
            #     (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size, log_path, datasource_path, tab)
            # )

            # ----------------------------------------------------
            # log_name = 'day_1_13_no'
            # tab = 'day_1_13_no'
            # datasource_path = '/home/lanbo/dataset/wifi_violence_processed_loc_no/'
            #
            # log_path = os.path.join('/home/lanbo/wifi_wavelet/log', log_name)
            # if not os.path.exists(log_path):
            #     os.makedirs(log_path)
            #
            # os.system(
            #     'bash /home/lanbo/wifi_wavelet/scripts/script_run.sh %d %s %s %s %s %d %s %s %s' %
            #     (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size, log_path, datasource_path, tab)
            # )
