import os
import logging
import pandas as pd
from torch.utils.data.dataset import Dataset

from data_process.dataset_config import DatasetConfig

from util import log_f_ch, load_mat

logger = logging.getLogger( )


class WiFiVioDatasetConfig(DatasetConfig):
    def __init__(self, datasource_path: os.path):
        super(WiFiVioDatasetConfig, self).__init__('wifiVio', datasource_path)


def load_wifi_Vio_data(config: WiFiVioDatasetConfig):

    train_data = {"data_path": os.path.join(config.datasource_path, 'train'),
                  "list_path": os.path.join(config.datasource_path, 'train_list.csv')}
    test_data = {"data_path": os.path.join(config.datasource_path, 'test'),
                  "list_path": os.path.join(config.datasource_path, 'test_list.csv')}

    return train_data, test_data

class WiFiVioDataset(Dataset):
    def __init__(self, data):
        super(WiFiVioDataset, self).__init__()

        logger.info('加载WiFiVio数据集')
        self.data_path = data['data_path']
        self.data_list = pd.read_csv(data['list_path'])
        self.tmp_data = self.data_list.iloc[0]['file']
        self.tmp_data = load_mat(os.path.join(data['data_path'], f'{self.tmp_data}.h5'))

        self.n_channel, self.seq_len = self.tmp_data['amp'].shape
        self.num_sample = len(self.data_list)

        self.label_n_class = 7
        self.freq_n_channel, self.freq_seq_len = None, None

        logger.info(log_f_ch('num_sample: ', str(self.num_sample)))
        logger.info(log_f_ch('n_class: ', str(self.label_n_class)))
        logger.info(log_f_ch('seq_len: ', str(self.seq_len)))
        logger.info(log_f_ch('n_channel: ', str(self.n_channel)))

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
    datasource_path = os.path.join("D:\study\dataset\wifi-partition-data-abs\dataset")
    # datasource_path = os.path.join("D:\study\dataset\wifi-partition-data-abs\dataset")

    train_data, test_data = load_wifi_Vio_data(WiFiVioDatasetConfig(datasource_path))
    train_dataset = WiFiVioDataset(train_data)
    test_dataset = WiFiVioDataset(test_data)
    print(len(train_dataset))
    print(len(test_dataset))
    print(train_dataset.get_n_classes(), test_dataset.get_n_classes())
    print(train_dataset.get_n_channels(), test_dataset.get_n_channels())
    print(train_dataset.get_seq_lens(), test_dataset.get_seq_lens())
