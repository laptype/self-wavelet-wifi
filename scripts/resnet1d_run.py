#!/usr/bin/env python3

import os
import sys
sys.path.append('/home/lanbo/wifi_wavelet')
from scripts.utils import *

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda = 1

    config = DatasetDefaultConfig()

    model_list = [
        ('resnet1d_101', 'resnet1d_span_cls_raw_time', 128),
        # ('resnet1d_101', 'resnet1d_span_cls_raw_channel', 128),
        # ('resnet1d_101', 'resnet1d_span_cls_freq_time', 128),
        # ('resnet1d_101', 'resnet1d_span_cls_freq_channel', 128),

        ('resnet1d_50', 'resnet1d_span_cls_raw_time', 128),
        # ('resnet1d_50', 'resnet1d_span_cls_raw_channel', 128),
        # ('resnet1d_50', 'resnet1d_span_cls_freq_time', 128),
        # ('resnet1d_50', 'resnet1d_span_cls_freq_channel', 128),

        ('resnet1d_34', 'resnet1d_span_cls_raw_time', 128),
        # ('resnet1d_34', 'resnet1d_span_cls_raw_channel', 128),
        # ('resnet1d_34', 'resnet1d_span_cls_freq_time', 128),
        # ('resnet1d_34', 'resnet1d_span_cls_freq_channel', 128),

        ('resnet1d_18', 'resnet1d_span_cls_raw_time', 128),
        # ('resnet1d_18', 'resnet1d_span_cls_raw_channel', 128),
        # ('resnet1d_18', 'resnet1d_span_cls_freq_time', 128),
        # ('resnet1d_18', 'resnet1d_span_cls_freq_channel', 128),
    ]
    config.dataset_list.append(f'WiVio')
    print(config.dataset_list)
    for dataset_name in config.dataset_list:
        for module in model_list:

            log_name = 'day_1_8'
            tab = 'day_1_8'
            datasource_path = '/home/lanbo/dataset/wifi_violence_processed/'

            log_path = os.path.join('/home/lanbo/wifi_wavelet/log', log_name)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]

            os.system(
                'bash /home/lanbo/wifi_wavelet/scripts/script_run.sh %d %s %s %s %s %d %s %s %s' %
                (cuda, dataset_name, backbone_name, head_name, strategy_name, batch_size, log_path, datasource_path, tab)
            )