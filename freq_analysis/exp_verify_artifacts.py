import sys
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 路径 Hack
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_utils import load_data_tntl
from utils.utils import setup_seed
from data_split import Cus_Dataset  # 必须引入

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

# 反归一化：将 Normalized Tensor 转回可观看的 RGB
def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

# ==========================================
# 核心修改：高斯软窗口 FDA 变换
# ==========================================
def gaussian_fda_transform(src_img, trg_img, beta=0.1):
    """
    使用高斯软窗口进行频谱混合，避免硬截断带来的振铃效应。
    src_img: (B, C, H, W) - 提供相位 (Content)
    trg_img: (B, C, H, W) - 提供低频幅度 (Style)
    beta: 控制高斯分布的 Sigma (Sigma = min(H,W) * beta)
    """
    # 1. FFT 和 Shift
    # clone() 防止修改原图
    fft_src = torch.fft.fftshift(torch.fft.fft2(src_img.clone()))
    fft_trg = torch.fft.fftshift(torch.fft.fft2(trg_img.clone()))

    # 2. 提取幅度谱和相位谱
    amp_src, pha_src = fft_src.abs(), fft_src.angle()
    amp_trg, _ = fft_trg.abs(), fft_trg.angle()

    # 3. 生成高斯掩码 (Gaussian Mask)
    B, C, H, W = src_img.shape
    device = src_img.device
    
    # 建立坐标网格，中心为 (0,0)
    y = torch.arange(H, device=device) - H // 2
    x = torch.arange(W, device=device) - W // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # 计算距离平方
    dist2 = X**2 + Y**2
    
    # 计算 Sigma
    # 这里定义 Sigma 与 beta 成正比。
    # 当 beta=0.1 时，Sigma 为图像尺度的 10%。
    # 高斯分布在 3*Sigma 处衰减接近 0，所以这覆盖了低频区域。
    sigma = max(1.0, min(H, W) * beta) 
    
    # 生成 Mask: 中心为 1 (取 Target)，边缘衰减为 0 (取 Source)
    # Mask shape: (H, W)
    mask = torch.exp(-dist2 / (2 * sigma**2))
    
    # 扩展 Mask 维度以匹配 Batch 和 Channel: (B, C, H, W)
    mask = mask.view(1, 1, H, W).expand(B, C, H, W)

    # 4. 软混合 (Soft Mixing)
    # Mixed = Target * Mask + Source * (1 - Mask)
    # 在低频中心，Mask接近1，主要取 Target (Style)
    # 在高频边缘，Mask接近0，主要取 Source (Detail/Content)
    amp_mixed = amp_trg * mask + amp_src * (1.0 - mask)

    # 5. 重建复数频谱
    fft_mixed = amp_mixed * torch.exp(1j * pha_src)

    # 6. Inverse FFT
    fft_mixed = torch.fft.ifftshift(fft_mixed)
    img_mixed = torch.fft.ifft2(fft_mixed)
    
    # 7. 取实部
    img_mixed = torch.real(img_mixed)
    
    return img_mixed

def run_visualization(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    
    print(f"Running Gaussian FDA Visualization: {args.domain_src} -> {args.domain_tgt}")
    
    # 1. 配置与数据加载
    config_dict = {
        'domain_src': args.domain_src,
        'domain_tgt': args.domain_tgt,
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'pre_split': True,
        'seed': args.seed,
        'device': device,
        'task_name': 'VizGaussian',
        'teacher_network': 'vgg11',
        'num_classes': 10
    }
    config = MockConfig(config_dict)
    
    try:
        (train_loaders, _, test_loaders, _) = load_data_tntl(config)
        src_loader = train_loaders[0]
        tgt_loader = test_loaders[1]
    except Exception as e:
        print(f"Data load error: {e}")
        return

    # 2. 抽取样本
    num_samples = 5
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    
    try:
        src_imgs_batch, _ = next(src_iter)
        tgt_imgs_batch, _ = next(tgt_iter)
    except StopIteration:
        return

    src_imgs = src_imgs_batch[:num_samples].to(device)
    tgt_imgs = tgt_imgs_batch[:num_samples].to(device)
    
    # 3. 定义 Beta 列表
    # 注意：高斯的 Sigma 计算方式不同，同样的 beta 值覆盖范围可能与矩形窗略有不同
    betas = [0.01, 0.05, 0.1, 0.2]
    
    # 4. 绘图
    num_cols = 2 + len(betas)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(3 * num_cols, 3 * num_samples))
    
    cols_title = ['Target (Content)', 'Source (Style Ref)'] + [f'Gaussian FDA (B={b})' for b in betas]
    for ax, col in zip(axes[0], cols_title):
        ax.set_title(col, fontsize=14, fontweight='bold')

    print("Generating Gaussian mixed images...")
    
    for i in range(num_samples):
        curr_tgt = tgt_imgs[i].unsqueeze(0)
        curr_src = src_imgs[i].unsqueeze(0)
        
        # Original
        axes[i, 0].imshow(denorm(curr_tgt[0]))
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(denorm(curr_src[0]))
        axes[i, 1].axis('off')
        
        # Mixed
        for j, beta in enumerate(betas):
            ax = axes[i, 2 + j]
            
            # === 调用新的高斯混合函数 ===
            mixed_img = gaussian_fda_transform(curr_tgt, curr_src, beta=beta)
            # =========================
            
            vis_img = denorm(mixed_img[0])
            ax.imshow(vis_img)
            ax.axis('off')

    plt.tight_layout()
    save_path = f"fda_gaussian_artifacts_{args.domain_src}_{args.domain_tgt}.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[Done] Result saved to: {save_path}")
    print("Check if the ringing artifacts (ripples) are gone compared to the previous result.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_src', type=str, default='cifar')
    parser.add_argument('--domain_tgt', type=str, default='stl')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2024)
    
    args = parser.parse_args()
    run_visualization(args)