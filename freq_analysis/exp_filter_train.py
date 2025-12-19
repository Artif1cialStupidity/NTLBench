import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tqdm import tqdm
import numpy as np

# 路径 Hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_utils import load_data_tntl, load_model
from utils.utils import setup_seed
from data_split import Cus_Dataset 
import freq_utils

# --- MMD Loss (内存优化版) ---
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        # 优化：使用 torch.cdist 计算平方欧氏距离
        # cdist 计算的是 p-norm，这里 p=2，然后再平方得到 L2_distance
        # 内存复杂度从 O(N^2 * D) 降低到 O(N^2)
        L2_distance = torch.cdist(total, total, p=2).pow(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        # 叠加多个高斯核
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss

# Mock Config
class MockConfig:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            setattr(self, k, v)
    def update(self, new_dict, allow_val_change=True):
        for k, v in new_dict.items():
            setattr(self, k, v)
    def keys(self):
        return self.__dict__.keys()

# --- 训练逻辑: Supervised Learning ---
def train_epoch_sl(model, loader, optimizer, criterion, device, filter_mode, radius):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training SL", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        if len(labels.shape) == 2: labels = torch.argmax(labels, dim=1)
        elif len(labels.shape) == 3: labels = torch.argmax(labels.squeeze(dim=1), dim=1)
            
        if filter_mode != 'none':
            with torch.no_grad():
                images = freq_utils.apply_frequency_filter(images, radius, mode=filter_mode)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

# --- 训练逻辑: NTL (Non-Transferable Learning) ---
def train_epoch_ntl(model, src_loader, tgt_loader, optimizer, criterion_kl, mmd_loss_fn, device, filter_mode, radius, args):
    model.train()
    running_loss = 0.0
    
    # 使用较短的长度，防止越界
    min_len = min(len(src_loader), len(tgt_loader))
    iter_src = iter(src_loader)
    iter_tgt = iter(tgt_loader)
    
    for _ in tqdm(range(min_len), desc="Training tNTL", leave=False):
        try:
            img1, label1 = next(iter_src)
            img2, label2 = next(iter_tgt)
        except StopIteration:
            break
            
        img1, label1 = img1.to(device).float(), label1.to(device).float()
        img2, label2 = img2.to(device).float(), label2.to(device).float()
        
        if filter_mode != 'none':
            with torch.no_grad():
                img1 = freq_utils.apply_frequency_filter(img1, radius, mode=filter_mode)
                img2 = freq_utils.apply_frequency_filter(img2, radius, mode=filter_mode)

        if args.network.startswith('vgg'):
            out1, out2, fe1, fe2 = model(img1, img2)
        else:
            if hasattr(model, 'forward_f'):
                out, feat = model.forward_f(torch.cat((img1, img2), 0))
            else:
                out, feat = model(torch.cat((img1, img2), 0), return_features=True)
            out1, out2 = out.chunk(2, dim=0)
            fe1, fe2 = feat.chunk(2, dim=0)

        out1 = F.log_softmax(out1, dim=1)
        loss1 = criterion_kl(out1, label1)

        out2 = F.log_softmax(out2, dim=1)
        loss2 = criterion_kl(out2, label2)

        # MMD Loss (使用优化后的类)
        mmd_loss_val = mmd_loss_fn(fe1.view(fe1.size(0), -1), fe2.view(fe2.size(0), -1)) * args.ntl_beta
        
        loss2_weighted = loss2 * args.ntl_alpha
        if loss2_weighted > 1: loss2_weighted = torch.clamp(loss2_weighted, 0, 1)
        
        mmd_clamped = mmd_loss_val
        if mmd_clamped > 1: mmd_clamped = torch.clamp(mmd_clamped, 0, 1)

        # tNTL Objective: 最小化 Source Loss，最大化 (Target Loss * MMD)
        loss = loss1 - loss2_weighted * mmd_clamped
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / min_len

# --- 通用评估逻辑 ---
def evaluate(model, loader, device, filter_mode, radius):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if len(labels.shape) == 2: labels = torch.argmax(labels, dim=1)
            elif len(labels.shape) == 3: labels = torch.argmax(labels.squeeze(dim=1), dim=1)
            
            if filter_mode != 'none':
                images = freq_utils.apply_frequency_filter(images, radius, mode=filter_mode)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    
    print(f"=== Experiment: {args.task_name} | {args.filter_mode.upper()}-Pass Filter (r={args.radius}) ===")
    
    config_dict = {
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
        'task_name': args.task_name,
        'pretrained_teacher': None,
        'surrogate_network': args.network, 
        'surrogate_pretrain': False,
        'NTL_alpha': args.ntl_alpha,
        'NTL_beta': args.ntl_beta
    }
    config = MockConfig(config_dict)
    
    try:
        (train_loaders, val_loaders, test_loaders, _) = load_data_tntl(config)
        src_train_loader = train_loaders[0]
        tgt_train_loader = train_loaders[1]
        src_test_loader = test_loaders[0]
        tgt_test_loader = test_loaders[1]
    except Exception as e:
        print(f"Data Error: {e}")
        import traceback; traceback.print_exc()
        return

    model = load_model(config)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    if args.task_name == 'tNTL':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        print("Using Adam optimizer for tNTL")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    mmd_loss_fn = MMD_loss().to(device)
    
    best_src_acc = 0.0
    
    for epoch in range(args.epochs):
        if args.task_name == 'SL':
            loss, train_acc = train_epoch_sl(
                model, src_train_loader, optimizer, criterion_ce, device, args.filter_mode, args.radius
            )
            log_str = f"Loss: {loss:.4f} | Train Acc: {train_acc:.2f}%"
            
        elif args.task_name == 'tNTL':
            loss = train_epoch_ntl(
                model, src_train_loader, tgt_train_loader, optimizer, criterion_kl, mmd_loss_fn, 
                device, args.filter_mode, args.radius, args
            )
            log_str = f"Loss: {loss:.4f}"
            
        scheduler.step()
        
        src_acc = evaluate(model, src_test_loader, device, args.filter_mode, args.radius)
        tgt_acc = evaluate(model, tgt_test_loader, device, args.filter_mode, args.radius)
        
        if src_acc > best_src_acc:
            best_src_acc = src_acc
            
        print(f"Epoch {epoch+1}/{args.epochs} | {log_str} | "
              f"Src Test: {src_acc:.2f}% | Tgt Test: {tgt_acc:.2f}%")
        
    print(f"\n[Final Result] {args.task_name} ({args.filter_mode}, r={args.radius})")
    print(f"Best Src Acc: {best_src_acc:.2f}%")
    print(f"Final Tgt Acc: {tgt_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='SL', choices=['SL', 'tNTL'])
    parser.add_argument('--domain_src', type=str, default='cifar')
    parser.add_argument('--domain_tgt', type=str, default='stl')
    parser.add_argument('--network', type=str, default='vgg11')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64) # 默认改为 64 防止 OOM
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2021)
    
    # Filter Params
    parser.add_argument('--filter_mode', type=str, default='none', choices=['none', 'low', 'high'])
    parser.add_argument('--radius', type=int, default=16)
    
    # NTL Params
    parser.add_argument('--ntl_alpha', type=float, default=0.1)
    parser.add_argument('--ntl_beta', type=float, default=0.1)
    
    args = parser.parse_args()
    run_experiment(args)