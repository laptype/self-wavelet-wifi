import os
import argparse
import logging
import torch

from config import BasicConfig, TrainConfig, TestConfig
from training import train
from testing import test

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from pipeline import Trainer
from config import TrainConfig


import init_util

logging.getLogger().setLevel(logging.DEBUG)

use_gpu = 1

class Config:
    dataset_name = 'WiAR_0.8'
    datasource_path = 'D:\study\postgraduate\study_project\wavelet_wifi\self-wavelet-wifi\dataset\WiAR'

    # backbone_name = 'wavevit_wave12_4_test_16'
    backbone_name = 'wavevit_wavelh2_4_test_16'
    head_name= 'wifi_ar_span_cls'
    strategy_name = 'vit_span_cls_raw'

    train_batch_size = 128

def _to_var(data: dict):
    if use_gpu:
        for key, value in data.items():
            data[key] = Variable(value.cuda())
    else:
        for key, value in data.items():
            data[key] = Variable(value)
    return data


def test(config):
    train_dataset, eval_dataset = init_util.init_dataset(config.dataset_name, config.datasource_path)

    strategy = init_util.init_strategy(config.backbone_name,
                                       config.head_name,
                                       config.strategy_name,
                                       train_dataset.get_n_channels(),
                                       train_dataset.get_seq_lens())

    train_data_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                   drop_last=True)
    # eval_data_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False),

    strategy = strategy.cuda()

    strategy.train()

    data = next(iter(train_data_loader))

    data = _to_var(data)

    loss = strategy(data)

    print(loss)

if __name__ == '__main__':
    test(Config)

