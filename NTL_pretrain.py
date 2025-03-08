import torch
import wandb
from utils.utils import *
from utils.load_utils import *
import pretrain
import os
import copy


if __name__ == '__main__':
    # Load config file from local
    wandb.init(project='NTLBenchmark', config='config/cifarstl/pretrain.yml')
    # wandb.init(project='NTLBenchmark', config='config/visda/pretrain.yml')
    
    config = wandb.config
    setup_seed(config.seed)
    wandbsweep_config_update(config)

    # load data
    (dataloader_train, dataloader_val, dataloader_test, datasets_name) = load_data_tntl(config)
    
    # load model
    model_ntl = load_model(config)
    model_ntl.eval()

    # pretrain
    if config.train_teacher_scratch:
        # set teacher
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
