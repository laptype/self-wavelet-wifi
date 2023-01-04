import os
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

from data_process.dataset_config import DatasetConfig
from .util import load_mat

logger = logging.getLogger( )


class WiFiARDatasetConfig(DatasetConfig):
    def __init__(self, datasource_path: os.path):
        super(WiFiARDatasetConfig, self).__init__('wifiVio', datasource_path)


def load_wifi_ar_data(config: WiFiARDatasetConfig):

    train_data = {"data_path": os.path.join(config.datasource_path, 'train'),
                  "list_path": os.path.join(config.datasource_path, 'train_list.csv')}
    test_data = {"data_path": os.path.join(config.datasource_path, 'test'),
                  "list_path": os.path.join(config.datasource_path, 'test_list.csv')}

    return train_data, test_data

class WiFiARDataset(Dataset):
    def __init__(self, data):
        super(WiFiARDataset, self).__init__()

        logger.info('加载WiFiAR数据集')
        self.data_path = data['data_path']
        self.data_list = pd.read_csv(data['list_path'])
        self.tmp_data = self.data_list.iloc[0]['file']
        self.tmp_data = load_mat(os.path.join(data['data_path'], f'{self.tmp_data}.h5'))

        self.n_channel, self.seq_len = self.tmp_data['amp'].shape
        self.num_sample = len(self.data_list)

        self.label_n_class = 7
        self.freq_n_channel, self.freq_seq_len = None, None

    def __getitem__(self, index):
        data = load_mat(os.path.join(self.data_path,
                                     f'{self.data_list.iloc[index]["file"]}.h5'))

        return {
            'data': data['amp'],
            # 'freq_data': self.freq_data[index],
            'label': data['label'],
        }

    def __len__(self):
        return self.num_sample

    def get_n_channels(self):
        return {
            'data': self.n_channel,
            'freq_data': self.freq_n_channel,
        }

    def get_seq_lens(self):
        return {
            'data': self.seq_len,
            'freq_data': self.freq_seq_len,
        }

    def get_n_classes(self):
        return {
            'label': self.label_n_class,
        }


if __name__ == '__main__':
    datasource_path = os.path.join("/home/wuxilei/data/wifi_har_empirical_study/wifi_ar")
    train_data, test_data = load_wifi_ar_data(WiFiARDatasetConfig(datasource_path, 0, 2, 0.75))
    train_dataset = WiFiARDataset(train_data)
    test_dataset = WiFiARDataset(test_data)
    print(len(train_dataset))
    print(len(test_dataset))
    print(train_dataset.get_n_classes(), test_dataset.get_n_classes())
    print(train_dataset.get_n_channels(), test_dataset.get_n_channels())
    print(train_dataset.get_seq_lens(), test_dataset.get_seq_lens())
