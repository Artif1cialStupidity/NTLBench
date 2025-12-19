import sys
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 路径 Hack: 确保能导入项目中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_utils import load_data_tntl, load_model
from utils.utils import setup_seed
from data_split import Cus_Dataset  # 必须引入，防止 torch.load 报错
import freq_utils  # 确保 freq_utils.py 在同一目录下

# Mock Config: 模拟 wandb 的配置对象
class MockConfig:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)
    def update(self, new_dict, allow_val_change=True):
        for k, v in new_dict.items():
            setattr(self, k, v)
    def keys(self):
        return self.__dict__.keys()

def evaluate_fda_attack(model, tgt_loader, src_loader_iter, device, betas, mode='low_swap'):
    """
    对指定模型执行 FDA 攻击并返回不同 beta 下的准确率列表
    """
    model.eval()
    acc_results = []
    
    print(f"--- Running FDA ({mode}) Attack ---")
    
    # 0. 先测 Baseline (Beta=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tgt_loader:
            images, labels = images.to(device), labels.to(device)
            if len(labels.shape) == 3:
                labels = torch.argmax(labels.squeeze(dim=1), dim=1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    base_acc = 100 * correct / total
    acc_results.append(base_acc)
    print(f"Beta: 0.00 (Clean), Acc: {base_acc:.2f}%")

    # 1. 遍历 Beta 进行攻击
    for beta in betas:
        correct = 0
        total = 0
        
        for tgt_imgs, tgt_labels in tqdm(tgt_loader, desc=f"Beta={beta}", leave=False):
            tgt_imgs, tgt_labels = tgt_imgs.to(device), tgt_labels.to(device)
            if len(tgt_labels.shape) == 3:
                tgt_labels = torch.argmax(tgt_labels.squeeze(dim=1), dim=1)
            
            # 获取随机的 Source Batch 作为风格参考
            try:
                src_imgs, _ = next(src_loader_iter)
            except StopIteration:
                # 迭代器耗尽，此处我们不重置，而是依赖外部传入的 cycle iterator 或者简单重新获取
                # 简单起见，这里假设 loader 足够长，或者在外部处理。
                # 为了稳健性，这里抛出异常让外层处理，或者直接 pass
                pass 
            
            # 截取相同大小
            if src_imgs.size(0) != tgt_imgs.size(0):
                min_bs = min(src_imgs.size(0), tgt_imgs.size(0))
                src_imgs = src_imgs[:min_bs]
                tgt_imgs = tgt_imgs[:min_bs]
                tgt_labels = tgt_labels[:min_bs]
            
            src_imgs = src_imgs.to(device)
            
            # === FDA 核心操作 ===
            # 将 Source 的低频(风格) 注入到 Target 图片中
            mixed_imgs = freq_utils.fda_transform(tgt_imgs, src_imgs, beta=beta, mode=mode)
            # ===================
            
            # 推理
            with torch.no_grad():
                outputs = model(mixed_imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += tgt_labels.size(0)
                correct += (predicted == tgt_labels).sum().item()
        
        acc = 100 * correct / total
        acc_results.append(acc)
        print(f"Beta: {beta:.2f}, Acc: {acc:.2f}%")
        
    return acc_results

def run_comparison(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    
    print(f"Experiment: {args.domain_src} -> {args.domain_tgt} using {args.network}")
    
    # --- 1. 统一配置 (用于加载数据) ---
    # 注意：数据加载与模型类型无关，只与域有关
    base_config_dict = {
        'domain_src': args.domain_src,
        'domain_tgt': args.domain_tgt,
        'teacher_network': args.network,
        'teacher_pretrain': False, 
        'num_classes': 10,
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'pre_split': True,
        'seed': args.seed,
        'device': device,
        'task_name': 'SL', # 仅用于初始化占位
        'pretrained_teacher': 'auto',
        'surrogate_network': args.network, 
        'surrogate_pretrain': False
    }
    config = MockConfig(base_config_dict)
    
    # --- 2. 加载数据 (只加载一次) ---
    print("Loading Data...")
    try:
        # train_loaders: [src, tgt], test_loaders: [src, tgt]
        (train_loaders, _, test_loaders, _) = load_data_tntl(config)
        
        # Source Train 用于提供 FDA 的幅度谱 (Style Reference)
        src_train_loader = train_loaders[0]
        # Target Test 用于评估模型性能
        tgt_test_loader = test_loaders[1]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 定义攻击参数
    betas = [0.01, 0.05, 0.1, 0.15, 0.2] # FDA 窗口大小
    plot_x = [0.0] + betas
    
    tasks = ['SL', 'tNTL']
    results = {}

    # --- 3. 循环评测模型 ---
    for task in tasks:
        print(f"\n========== Evaluating Model: {task} ==========")
        
        # 3.1 构造模型路径
        if task == 'SL':
            # SL 模型路径通常是 SL_{src}_{net}.pth
            model_name = f'SL_{args.domain_src}_{args.network}.pth'
        else:
            # tNTL 模型路径通常是 tNTL_{src}_{tgt}_{net}.pth
            model_name = f'tNTL_{args.domain_src}_{args.domain_tgt}_{args.network}.pth'
            
        model_path = os.path.join('./saved_models', model_name)
        
        if not os.path.exists(model_path):
            print(f"[Warning] Model file not found: {model_path}")
            print(f"Skipping {task} evaluation. Please pre-train it first.")
            results[task] = None
            continue
            
        # 3.2 加载模型结构与权重
        # 更新 config 中的 task_name，以防 load_model 内部有特定逻辑
        config.task_name = task 
        try:
            model = load_model(config)
            print(f"Loading weights from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model {task}: {e}")
            results[task] = None
            continue

        # 3.3 运行攻击
        # 创建一个无限循环的 source 迭代器，保证 target 每个 batch 都能取到对应的 source 图片
        def infinite_iter(loader):
            while True:
                for batch in loader:
                    yield batch
        src_iter = infinite_iter(src_train_loader)
        
        # 执行 low_swap (标准风格迁移攻击)
        accs = evaluate_fda_attack(model, tgt_test_loader, src_iter, device, betas, mode='high_swap')
        results[task] = accs

    # --- 4. 绘图对比 ---
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))
    
    # 绘制 SL (Baseline)
    if results['SL'] is not None:
        plt.plot(plot_x, results['SL'], marker='o', color='red', 
                 linewidth=2, label='SL (Baseline) - Vulnerable')
        # 标注 Baseline 的变化幅度
        delta_sl = results['SL'][-1] - results['SL'][0]
        print(f"SL Accuracy Change: {results['SL'][0]:.2f}% -> {results['SL'][-1]:.2f}% (Delta: {delta_sl:+.2f}%)")

    # 绘制 tNTL (Ours)
    if results['tNTL'] is not None:
        plt.plot(plot_x, results['tNTL'], marker='s', color='green', 
                 linewidth=2, label='tNTL (Ours) - Robust')
        delta_tntl = results['tNTL'][-1] - results['tNTL'][0]
        print(f"tNTL Accuracy Change: {results['tNTL'][0]:.2f}% -> {results['tNTL'][-1]:.2f}% (Delta: {delta_tntl:+.2f}%)")

    plt.title(f"Robustness against FDA Style Attack: {args.domain_src} -> {args.domain_tgt}")
    plt.xlabel("FDA Beta (Style Transfer Intensity)")
    plt.ylabel("Target Accuracy (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    save_path = f"fda_compare_{args.domain_src}_{args.domain_tgt}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nComparison plot saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_src', type=str, default='cifar')
    parser.add_argument('--domain_tgt', type=str, default='stl')
    parser.add_argument('--network', type=str, default='vgg11') # 确保与你保存的模型一致
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2021)
    
    args = parser.parse_args()
    run_comparison(args)