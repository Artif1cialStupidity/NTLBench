import sys
import os
import torch
import argparse
from tqdm import tqdm

# 将父目录加入 path 以便导入 utils 和 data_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_utils import load_data_tntl
from utils.utils import setup_seed
import freq_utils

# ---------------------------------------------------------
# 【关键修复】: 将 Cus_Dataset 引入当前命名空间
# 这步是必须的，因为 torch.load 需要在 __main__ 中找到这个类定义
# ---------------------------------------------------------
from data_split import Cus_Dataset 

# 模拟 wandb.config 的行为
class MockConfig:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)
    
    def update(self, new_dict, allow_val_change=True):
        for k, v in new_dict.items():
            setattr(self, k, v)
    
    def keys(self):
        return self.__dict__.keys()

def run_analysis(args):
    # 1. 配置环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    
    config_dict = {
        'domain_src': args.domain_src,
        'domain_tgt': args.domain_tgt,
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'pre_split': True,
        'seed': args.seed,
        'device': device,
        'task_name': 'FreqAnalysis'
    }
    
    config = MockConfig(config_dict)
    
    # 2. 加载数据
    try:
        (dataloader_train, _, _, _) = load_data_tntl(config)
        src_loader = dataloader_train[0]
        tgt_loader = dataloader_train[1] 
    except Exception as e:
        print(f"Data loading failed: {e}")
        # 打印更详细的错误栈以便调试
        import traceback
        traceback.print_exc()
        return

    print(f"Analyzing Gap between Source: {args.domain_src} and Target: {args.domain_tgt}")
    
    # 3. 计算 Source 的平均幅度谱
    src_amp_sum = None
    src_count = 0
    
    print("Processing Source Domain...")
    for imgs, _ in tqdm(src_loader):
        imgs = imgs.to(device)
        amp, _ = freq_utils.calc_fft(imgs) 
        
        batch_sum = torch.sum(amp, dim=0)
        if src_amp_sum is None:
            src_amp_sum = batch_sum
        else:
            src_amp_sum += batch_sum
        src_count += imgs.shape[0]
        
    src_amp_avg = src_amp_sum / src_count
    
    # 4. 计算 Target 的平均幅度谱
    tgt_amp_sum = None
    tgt_count = 0
    
    print("Processing Target Domain...")
    for imgs, _ in tqdm(tgt_loader):
        imgs = imgs.to(device)
        amp, _ = freq_utils.calc_fft(imgs)
        
        batch_sum = torch.sum(amp, dim=0)
        if tgt_amp_sum is None:
            tgt_amp_sum = batch_sum
        else:
            tgt_amp_sum += batch_sum
        tgt_count += imgs.shape[0]
        
    tgt_amp_avg = tgt_amp_sum / tgt_count
    
    # 5. 计算差异
    diff_amp = torch.abs(src_amp_avg - tgt_amp_avg)
    
    # 6. 可视化并保存
    save_name = f"spectral_analysis_{args.domain_src}_to_{args.domain_tgt}.png"
    freq_utils.plot_spectral_analysis(src_amp_avg, tgt_amp_avg, diff_amp, save_path=save_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_src', type=str, default='cifar', help='Source domain')
    parser.add_argument('--domain_tgt', type=str, default='stl', help='Target domain')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2021)
    
    args = parser.parse_args()
    run_analysis(args)