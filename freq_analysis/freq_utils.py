import torch
import numpy as np
import matplotlib.pyplot as plt

def calc_fft(tensor_imgs):
    """
    输入: tensor (B, C, H, W)
    输出: 幅度谱 (B, C, H, W), 相位谱 (B, C, H, W)
    注意: 输出的幅度谱已经经过 log 变换和 fftshift (低频移至中心)
    """
    # 1. FFT 变换
    fft = torch.fft.fft2(tensor_imgs)
    
    # 2. Shift: 将低频分量移到图像中心
    fshift = torch.fft.fftshift(fft)
    
    # 3. 计算幅度 (Amplitude) 和 相位 (Phase)
    # 使用 log(abs + epsilon) 来压缩动态范围，便于可视化和统计
    amp = torch.log(torch.abs(fshift) + 1e-8)
    phase = torch.angle(fshift)
    
    return amp, phase

def get_radial_profile(img_2d):
    """
    计算 2D 图像的径向平均分布 (Azimuthal Average)。
    将 2D 频谱压缩为 1D 曲线: X轴为频率半径(低频->高频), Y轴为能量/差异。
    
    输入: 2D numpy array (H, W)
    输出: 1D numpy array (Radius_Max)
    """
    h, w = img_2d.shape
    center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    
    # 计算每个像素到中心的距离 r
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # 统计每个半径 r 上的平均值
    tbin = np.bincount(r.ravel(), img_2d.ravel())
    nr = np.bincount(r.ravel())
    
    # 避免除以零
    radialprofile = tbin / (nr + 1e-8)
    
    # 只取有效半径范围 (即 min(h, w) // 2)
    max_radius = min(h, w) // 2
    return radialprofile[:max_radius]

def plot_spectral_analysis(src_amp_avg, tgt_amp_avg, diff_amp, save_path=None):
    """
    绘制分析图: Source谱, Target谱, 差值谱, 1D频段差异曲线
    """
    # 转换为 Numpy 并取均值 (C, H, W) -> (H, W) 用于灰度展示
    src_np = torch.mean(src_amp_avg, dim=0).cpu().numpy()
    tgt_np = torch.mean(tgt_amp_avg, dim=0).cpu().numpy()
    diff_np = torch.mean(diff_amp, dim=0).cpu().numpy()
    
    # 计算径向分布 (1D Curve)
    radial_diff = get_radial_profile(diff_np)
    freqs = np.arange(len(radial_diff))
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Source Spectrum
    im0 = axes[0].imshow(src_np, cmap='inferno')
    axes[0].set_title("Source Avg Log-Amp Spectrum")
    plt.colorbar(im0, ax=axes[0])
    
    # 2. Target Spectrum
    im1 = axes[1].imshow(tgt_np, cmap='inferno')
    axes[1].set_title("Target Avg Log-Amp Spectrum")
    plt.colorbar(im1, ax=axes[1])
    
    # 3. Difference Map (Source - Target)
    im2 = axes[2].imshow(diff_np, cmap='jet')
    axes[2].set_title("Spectral Difference (L1 dist)")
    plt.colorbar(im2, ax=axes[2])
    
    # 4. 1D Frequency Sensitivity Curve
    axes[3].plot(freqs, radial_diff, color='red', linewidth=2)
    axes[3].set_title("Domain Gap vs. Frequency")
    axes[3].set_xlabel("Frequency Radius (Low -> High)")
    axes[3].set_ylabel("Mean Amplitude Difference")
    axes[3].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Result saved to {save_path}")
    else:
        plt.show()

# ... (保留之前的 imports 和 函数)

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxw
    fft_amp = fft_im.abs()
    fft_pha = fft_im.angle()
    return fft_amp, fft_pha

def low_freq_mutate(amp_src, amp_trg, beta=0.1):
    """
    Standard FDA: 将 Target 的低频幅度 (Center) 替换给 Source
    """
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h,w))*beta)).astype(int)
    
    # 也就是保留 amp_src 的高频，使用 amp_trg 的低频
    amp_src[:, :, int(h/2)-b:int(h/2)+b+1, int(w/2)-b:int(w/2)+b+1] = \
        amp_trg[:, :, int(h/2)-b:int(h/2)+b+1, int(w/2)-b:int(w/2)+b+1]
    
    return amp_src

def high_freq_mutate(amp_src, amp_trg, beta=0.1):
    """
    Inverse FDA: 将 Target 的高频幅度 (Border) 替换给 Source
    保留 Source 的低频 (Center)
    """
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h,w))*beta)).astype(int)
    
    # 创建一个全 Target 的底板
    amp_new = amp_trg.clone()
    
    # 将 Source 的低频 (Center) 覆盖回去
    # 结果 = Source低频 + Target高频
    amp_new[:, :, int(h/2)-b:int(h/2)+b+1, int(w/2)-b:int(w/2)+b+1] = \
        amp_src[:, :, int(h/2)-b:int(h/2)+b+1, int(w/2)-b:int(w/2)+b+1]
        
    return amp_new

def fda_transform(src_img, trg_img, beta=0.1, mode='low_swap'):
    """
    输入: 
        src_img: 内容图像 (提供相位和部分幅度)
        trg_img: 风格图像 (提供部分幅度)
        beta: 窗口比例 (0.0 ~ 1.0)
        mode: 
            'low_swap': 将 trg 的低频给 src (标准 FDA, 模拟风格迁移)
            'high_swap': 将 trg 的高频给 src (修复细节差异)
    输出:
        mixed_img: 变换后的图像
    """
    # 1. FFT
    fft_src = torch.fft.fft2(src_img.clone())
    fft_trg = torch.fft.fft2(trg_img.clone())

    # 2. Shift (将低频移到中心，便于操作)
    fft_src = torch.fft.fftshift(fft_src)
    fft_trg = torch.fft.fftshift(fft_trg)

    # 3. 提取幅度谱和相位谱
    amp_src, pha_src = extract_ampl_phase(fft_src)
    amp_trg, _ = extract_ampl_phase(fft_trg)

    # 4. 幅度谱混合
    if mode == 'low_swap':
        # Src 内容 + Trg 低频 (Style)
        amp_mixed = low_freq_mutate(amp_src.clone(), amp_trg.clone(), beta=beta)
    elif mode == 'high_swap':
        # Src 内容 + Trg 高频 (Detail)
        amp_mixed = high_freq_mutate(amp_src.clone(), amp_trg.clone(), beta=beta)
    else:
        raise ValueError("mode must be 'low_swap' or 'high_swap'")

    # 5. 重建复数频谱: Amplitude * exp(i * Phase)
    fft_mixed = amp_mixed * torch.exp(1j * pha_src)

    # 6. Inverse Shift
    fft_mixed = torch.fft.ifftshift(fft_mixed)

    # 7. Inverse FFT
    img_mixed = torch.fft.ifft2(fft_mixed)
    
    # 8. 取实部并截断到合理范围
    img_mixed = torch.real(img_mixed)
    # img_mixed = torch.clamp(img_mixed, 0, 1) # 注意：如果是 Normalize 后的数据，范围不是 0-1
    
    return img_mixed

# ... (保留之前的代码)

def get_circular_mask(h, w, radius, type='low'):
    """
    生成理想的圆形掩膜 (Ideal Circular Mask)
    """
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    if type == 'low':
        # 低通：中心为1，外围为0
        mask = dist_from_center <= radius
    elif type == 'high':
        # 高通：中心为0，外围为1
        mask = dist_from_center > radius
    else:
        raise ValueError("Type must be 'low' or 'high'")
    
    return torch.from_numpy(mask).float()

def apply_frequency_filter(imgs, radius, mode='low'):
    """
    对 Batch 图片应用频域滤波器
    imgs: (B, C, H, W)
    radius: 截止频率半径
    mode: 'low' (Low-pass) or 'high' (High-pass)
    """
    device = imgs.device
    b, c, h, w = imgs.shape
    
    # 1. FFT
    fft = torch.fft.fft2(imgs)
    fshift = torch.fft.fftshift(fft)
    
    # 2. 生成 Mask (广播到 Batch 和 Channel)
    mask = get_circular_mask(h, w, radius, type=mode).to(device)
    mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    
    # 3. 应用 Mask
    fshift_filtered = fshift * mask
    
    # 4. iFFT
    fft_filtered = torch.fft.ifftshift(fshift_filtered)
    img_filtered = torch.fft.ifft2(fft_filtered)
    
    # 5. 取实部
    img_filtered = torch.real(img_filtered)
    
    return img_filtered