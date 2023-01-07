import os
import logging
import torch

from torch.utils.data.dataloader import DataLoader

from pipeline import Trainer, Tester
from config import TrainConfig

import init_util
from util import misc

logger = logging.getLogger(__name__)


def train(config: TrainConfig):
    if config.gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device



    train_dataset, eval_dataset = init_util.init_dataset(config.dataset_name, config.datasource_path)
    strategy = init_util.init_strategy(config.backbone_name,
                                       config.head_name,
                                       config.strategy_name,
                                       train_dataset.get_n_channels(),
                                       train_dataset.get_seq_lens())

    # ----------------------------- TRAIN---------------------------

    trainer = Trainer(
        strategy=strategy,
        train_data_loader=DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                     drop_last=True),
        eval_data_loader=DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False),
        num_epoch=config.num_epoch,
        opt_method=config.opt_method,
        lr_rate=config.lr_rate,
        lr_rate_adjust_epoch=config.lr_rate_adjust_epoch,
        lr_rate_adjust_factor=config.lr_rate_adjust_factor,
        weight_decay=config.weight_decay,
        save_epoch=config.save_epoch,
        eval_epoch=config.eval_epoch,
        patience=config.patience,
        check_point_path=config.check_point_path,
        use_gpu=False if config.gpu_device is None else True,
    )

    trainer.training()

    # ----------------------------- TEST---------------------------
    print('TEST-final'.center(100, '='))

    strategy.load_state_dict(torch.load(os.path.join(config.check_point_path, "%s-%s-final" % (
        config.backbone_name, config.head_name,
    ))))

    tester = Tester(
        strategy=strategy,
        eval_data_loader=DataLoader(eval_dataset, batch_size=config.train_batch_size, shuffle=False),
        n_classes=eval_dataset.get_n_classes(),
        output_path=config.check_point_path,
        use_gpu=False if config.gpu_device is None else True,
    )

    tester.testing()

    # ----------------------------- TEST---------------------------
    print('TEST-best'.center(100, '='))

    strategy.load_state_dict(torch.load(os.path.join(config.check_point_path, "%s-%s-best" % (
        config.backbone_name, config.head_name,
    ))))

    tester = Tester(
        strategy=strategy,
        eval_data_loader=DataLoader(eval_dataset, batch_size=config.train_batch_size, shuffle=False),
        n_classes=eval_dataset.get_n_classes(),
        output_path=config.check_point_path,
        use_gpu=False if config.gpu_device is None else True,
    )

    tester.testing()