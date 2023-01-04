import os

from scripts.utils import *


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda = 0

    config = DatasetDefaultConfig()

    model_list = [
        # ('vit_b_2', 'vit_span_cls_freq', 128),
        # ('vit_b_4', 'vit_span_cls_freq', 128),
        # ('vit_b_8', 'vit_span_cls_freq', 128),
        ('vit_b_16', 'vit_span_cls_raw', 128),
        ('vit_b_32', 'vit_span_cls_raw', 128),
        ('vit_b_64', 'vit_span_cls_raw', 128),
        #
        # ('vit_s_2', 'vit_span_cls_freq', 128),
        # ('vit_s_4', 'vit_span_cls_freq', 128),
        # ('vit_s_8', 'vit_span_cls_freq', 128),
        # ('vit_s_16', 'vit_span_cls_raw', 128),
        # ('vit_s_32', 'vit_span_cls_raw', 128),
        # ('vit_s_64', 'vit_span_cls_raw', 128),
        #
        # ('vit_ms_2', 'vit_span_cls_freq', 128),
        # ('vit_ms_4', 'vit_span_cls_freq', 128),
        # ('vit_ms_8', 'vit_span_cls_freq', 128),
        # ('vit_ms_16', 'vit_span_cls_raw', 128),
        # ('vit_ms_32', 'vit_span_cls_raw', 128),
        # ('vit_ms_64', 'vit_span_cls_raw', 128),
        #
        # ('vit_es_2', 'vit_span_cls_freq', 128),
        # ('vit_es_4', 'vit_span_cls_freq', 128),
        # ('vit_es_8', 'vit_span_cls_freq', 128),
        # ('vit_es_16', 'vit_span_cls_raw', 128),
        # ('vit_es_32', 'vit_span_cls_raw', 128),
        # ('vit_es_64', 'vit_span_cls_raw', 128),
    ]
    config.dataset_list.append(f'WiVio')
    print(config.dataset_list)
    for dataset_name in config.dataset_list:
        for module in model_list:
            datasource_path = 'D:\study\dataset\wifi-partition-data-abs\dataset'
            backbone_name = module[0]
            head_name = dataset_name_to_head_name_mapping(dataset_name)
            strategy_name = module[1]
            batch_size = module[2]

            eval_batch_size = 1
            num_epoch = 500

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
            os.chdir("D:\study\postgraduate\study_project\wavelet_wifi\self-wavelet-wifi")
            os.system('conda activate test')
            os.system(
                f'python main.py --dataset_name "{dataset_name}" --gpu_device {cuda} \
                --backbone_name "{backbone_name}" --head_name "{head_name}" --strategy_name "{strategy_name}" \
                --train_batch_size {train_batch_size} --eval_batch_size {eval_batch_size} --num_epoch {num_epoch} \
                --opt_method {opt_method} --lr_rate {lr_rate} --weight_decay {weight_decay} --datasource_path {datasource_path}\
                --lr_rate_adjust_epoch {lr_rate_adjust_epoch} --lr_rate_adjust_factor {lr_rate_adjust_factor}  \
                --save_epoch {save_epoch} --eval_epoch {eval_epoch} --patience {patience} --is_train true \
                > {dataset_name}-{strategy_name}-TRAIN.log'
            )
            # os.system(f'PYTHON main.py --dataset_name "{dataset_name}" --gpu_device {cuda} \
            #             --backbone_name "{backbone_name}" --head_name "{head_name}" --strategy_name "{strategy_name}" \
            #             --test_batch_size {test_batch_size} \
            #             > {dataset_name}-{strategy_name}-TEST.log')
