import torch
import wandb
from utils.utils import *
from utils.load_utils import *
import pretrain
import os
import copy
import argparse

if __name__ == '__main__':
    # 1. 初始化命令行参数解析器
    parser = argparse.ArgumentParser(description='NTL Pretrain')
    
    # 定义命令行参数
    parser.add_argument('--task_name', type=str, default=None, help='Task name (SL, tNTL, etc.)')
    parser.add_argument('--domain_src', type=str, default=None, help='Source domain')
    parser.add_argument('--domain_tgt', type=str, default=None, help='Target domain') # 建议加上 target，方便命名
    parser.add_argument('--teacher_network', type=str, default=None, help='Network architecture')
    parser.add_argument('--pretrain_epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--image_size', type=int, default=None, help='Image size')
    parser.add_argument('--config', type=str, default='config/cifarstl/pretrain.yml', help='Path to config file')

    args = parser.parse_args()

    # [优化点]：动态生成 WandB 的 run name，方便区分实验
    # 如果没传参数，就用默认的；传了就组合一下
    run_name = None
    if args.task_name and args.domain_src:
        tgt_suffix = f"-{args.domain_tgt}" if args.domain_tgt else ""
        run_name = f"{args.task_name}-{args.domain_src}{tgt_suffix}"

    # 2. 初始化 wandb
    wandb.init(project='NTLBenchmark', config=args.config, name=run_name)
    
    config = wandb.config

    # 3. 使用命令行参数覆盖配置
    if args.task_name is not None:
        config.update({'task_name': args.task_name}, allow_val_change=True)
    if args.domain_src is not None:
        config.update({'domain_src': args.domain_src}, allow_val_change=True)
    if args.domain_tgt is not None: # 别忘了更新 target
        config.update({'domain_tgt': args.domain_tgt}, allow_val_change=True)
    if args.teacher_network is not None:
        config.update({'teacher_network': args.teacher_network}, allow_val_change=True)
    if args.pretrain_epochs is not None:
        config.update({'pretrain_epochs': args.pretrain_epochs}, allow_val_change=True)
    if args.image_size is not None:
        config.update({'image_size': args.image_size}, allow_val_change=True)

    # 4. 后续逻辑保持不变
    setup_seed(config.seed)
    wandbsweep_config_update(config)

    # load data
    (dataloader_train, dataloader_val, dataloader_test, datasets_name) = load_data_tntl(config)
    
    # load model
    model_ntl = load_model(config)
    model_ntl.eval()

    # pretrain
    if config.train_teacher_scratch:
        cprint('train model from scratch', 'magenta')
        cprint(f'method: {config.task_name}', 'yellow')
        if config.task_name in ['SL']:
            trainer_func = pretrain.trainer_SL.train_src
        elif config.task_name in ['tNTL', 'sNTL']:
            trainer_func = pretrain.trainer_NTL.train_tntl
        elif config.task_name in ['tCUTI', 'sCUTI']:
            trainer_func = pretrain.trainer_CUTI.train_tCUTI
        elif config.task_name in ['tCUPI']:
            trainer_func = pretrain.trainer_CUPI.train_tCUPI
        elif config.task_name in ['tHNTL']:
            trainer_func = pretrain.trainer_HNTL.train_tHNTL
        elif config.task_name in ['tSOPHON']:
            trainer_func = pretrain.trainer_SOPHON.train_sophon
        else:
            raise NotImplementedError
        trainer_func(config, dataloader_train, dataloader_val, dataloader_test,
                     model_ntl, datasets_name=datasets_name)
        
        if config.save_train_teacher:    
            if config.pretrained_teacher == 'auto':
                save_path = auto_save_name(config)
            else: 
                save_path = config.pretrained_teacher
            cprint(f'save path: {save_path}')
            torch.save(model_ntl.state_dict(), save_path)