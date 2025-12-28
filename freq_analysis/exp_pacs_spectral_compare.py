import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# --- 路径 Hack: 确保能导入 utils ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ntl_utils import utils_pacs
from freq_analysis import freq_utils

# ==========================================
# 辅助函数: 计算数据集的平均 Radial PSD
# ==========================================
def get_average_radial_profile(data_loader_func, domain_name, max_images=1000, image_size=224):
    print(f"Processing {domain_name}...")
    
    # 获取原始数据 (List of arrays, Labels, Size)
    # utils_pacs 返回的是 [list_img, list_label, data_size]
    # list_img 是 numpy array (N, 224, 224, 3) RGB
    try:
        raw_data = data_loader_func()
    except Exception as e:
        print(f"Error loading {domain_name}: {e}")
        return None

    images = raw_data[0]
    if len(images) > max_images:
        # 随机采样以加快计算
        indices = np.random.choice(len(images), max_images, replace=False)
        images = images[indices]
    
    psd_accum = None
    count = 0
    
    for i in tqdm(range(len(images)), desc=f"{domain_name}"):
        # 预处理: Numpy (H,W,C) -> Tensor (1, C, H, W)
        img = images[i]
        
        # 确保 resize 到统一大小 (以防 utils_pacs 没做)
        if img.shape[0] != image_size:
            img = cv2.resize(img, (image_size, image_size))
            
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        
        # 计算 FFT 幅度谱
        # calc_fft 返回 (Amp, Phase), Amp 已经是 log(abs + epsilon)
        # 我们这里需要原始能量用于物理意义的 PSD，或者直接用 Log-Magnitude 也可以
        # 为了对比能量分布，通常使用 Magnitude (未 Log) 或者 Log-Magnitude
        # 这里为了视觉清晰，我们使用 freq_utils.calc_fft 返回的 Log-Magnitude
        amp, _ = freq_utils.calc_fft(img_tensor) 
        
        # 取 Channel 平均 (灰度化频谱) -> (1, H, W)
        amp_avg = torch.mean(amp, dim=1).squeeze(0).cpu().numpy()
        
        # 计算径向分布 (Radial Profile)
        profile = freq_utils.get_radial_profile(amp_avg)
        
        if psd_accum is None:
            psd_accum = np.zeros_like(profile)
        
        # 对齐长度 (防止 resize 误差)
        min_len = min(len(psd_accum), len(profile))
        psd_accum[:min_len] += profile[:min_len]
        count += 1
        
    return psd_accum / count

# ==========================================
# 主程序
# ==========================================
def main():
    # 1. 定义数据源
    domains = {
        'Photo (P)': utils_pacs.get_pacs_P,
        'Art (A)': utils_pacs.get_pacs_A,
        'Cartoon (C)': utils_pacs.get_pacs_C,
        'Sketch (S)': utils_pacs.get_pacs_S
    }
    
    # 颜色映射
    colors = {
        'Photo (P)': 'tab:blue', 
        'Art (A)': 'tab:orange', 
        'Cartoon (C)': 'tab:green', 
        'Sketch (S)': 'tab:red'
    }
    
    profiles = {}
    
    # 2. 计算各域 PSD
    for name, func in domains.items():
        profile = get_average_radial_profile(func, name, max_images=500)
        if profile is not None:
            profiles[name] = profile

    # 3. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Plot 1: Radial PSD (Log Power vs Frequency) ---
    ax_psd = axes[0]
    for name, profile in profiles.items():
        freqs = np.arange(len(profile))
        # Normalize to 0-1 for better comparison shape
        norm_profile = (profile - profile.min()) / (profile.max() - profile.min())
        
        ax_psd.plot(freqs, norm_profile, label=name, color=colors[name], linewidth=2)
        
    ax_psd.set_title("Radial Power Spectral Density (PSD)", fontsize=14)
    ax_psd.set_xlabel("Frequency Radius (Low -> High)", fontsize=12)
    ax_psd.set_ylabel("Normalized Log-Energy", fontsize=12)
    ax_psd.grid(True, linestyle='--', alpha=0.5)
    ax_psd.legend(fontsize=10)
    
    # --- Plot 2: Cumulative Energy Distribution (CDF) ---
    ax_cdf = axes[1]
    for name, profile in profiles.items():
        # 反转 Log 变换以获得真实能量 (近似) 用于 CDF 计算
        # 因为 profile 是 log scale，直接累加 log 值物理意义不明确
        # 但如果仅仅比较“分布集中度”，直接累加 profile 也能体现趋势
        # 这里我们尝试恢复线性能量: exp(profile)
        linear_energy = np.exp(profile) 
        
        cdf = np.cumsum(linear_energy)
        cdf_norm = cdf / cdf[-1] # 归一化到 0-1
        
        freqs = np.arange(len(cdf_norm))
        ax_cdf.plot(freqs, cdf_norm, label=name, color=colors[name], linewidth=2)

        # 标注 90% 能量点
        idx_90 = np.argmax(cdf_norm > 0.9)
        ax_cdf.scatter(freqs[idx_90], 0.9, color=colors[name], s=30)
        
    ax_cdf.set_title("Cumulative Energy Distribution (CDF)", fontsize=14)
    ax_cdf.set_xlabel("Frequency Radius (Low -> High)", fontsize=12)
    ax_cdf.set_ylabel("Cumulative Energy Proportion", fontsize=12)
    ax_cdf.grid(True, linestyle='--', alpha=0.5)
    ax_cdf.legend(fontsize=10)
    ax_cdf.axhline(0.9, color='gray', linestyle=':', alpha=0.5, label='90% Energy')

    # 保存
    save_path = 'freq_analysis/pacs_spectral_analysis.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nAnalysis saved to {save_path}")
    # plt.show()

if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    main()