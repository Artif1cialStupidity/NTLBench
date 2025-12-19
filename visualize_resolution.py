import sys
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 使得可以导入项目模块
sys.path.append(os.path.abspath('./'))

# 导入数据读取工具
import utils.ntl_utils.utils_pacs as utils_pacs
import utils.ntl_utils.utils_digit as utils_digit

def visualize_comparison():
    # 1. 获取数据
    # PACS 原始加载逻辑是 Resize 到 224 (参见 utils_pacs.py)
    # Digits 原始加载逻辑是 Resize 到 32 (参见 utils_digit.py)
    print("Loading one sample from PACS and Digits...")
    
    # 获取一张 PACS Photo 图片
    # 注意：get_pacs_P 返回 [list_img, list_label, size]
    # list_img 中的图片已经是 224x224 的 numpy 数组
    pacs_data = utils_pacs.get_pacs_P()
    img_pacs_original = pacs_data[0][0]  # 取第一张图 (224x224)
    
    # 获取一张 Digits (MNIST) 图片
    # list_img 中的图片是 32x32
    digits_data = utils_digit.get_mnist_rotate_0()
    img_digits_original = digits_data[0][0] # 取第一张图 (32x32)

    # 2. 模拟 64x64 的训练环境
    # 定义 Resize 变换
    resize_64 = transforms.Resize((64, 64))
    to_pil = transforms.ToPILImage()
    
    # 处理 PACS: 224 -> 64
    # 模拟训练时的数据流：先转 PIL -> Resize 64 -> (训练用) -> Resize 回 224 (为了方便人眼对比大小，或者直接显示小图)
    # 这里我们直接显示 64 的图，利用 matplotlib 放大看像素化程度
    img_pacs_64 = cv2.resize(img_pacs_original, (64, 64), interpolation=cv2.INTER_LINEAR)
    
    # 处理 Digits: 32 -> 64 (Upsample)
    img_digits_64 = cv2.resize(img_digits_original, (64, 64), interpolation=cv2.INTER_LINEAR)

    # 3. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # --- Row 1: PACS ---
    # Original (224)
    axes[0, 0].imshow(img_pacs_original)
    axes[0, 0].set_title(f"PACS Original (224x224)\nFeature Map: 7x7")
    axes[0, 0].axis('off')
    
    # Resized (64)
    axes[0, 1].imshow(img_pacs_64)
    axes[0, 1].set_title(f"PACS Training (64x64)\nFeature Map: 2x2 (Too Small!)")
    axes[0, 1].axis('off')
    
    # --- Row 2: Digits ---
    # Original (32)
    axes[1, 0].imshow(img_digits_original)
    axes[1, 0].set_title(f"Digits Original (32x32)\nFeature Map: 1x1")
    axes[1, 0].axis('off')
    
    # Resized (64)
    axes[1, 1].imshow(img_digits_64)
    axes[1, 1].set_title(f"Digits Training (64x64)\nUpsampled (No Loss)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = 'resolution_check.png'
    plt.savefig(save_path)
    print(f"\nVisualization saved to {save_path}")
    print("-" * 50)
    print("ANALYSIS:")
    print("1. PACS (Row 1): Compare the details. At 64x64, fine textures are gone.")
    print("   More importantly, for VGG13, input 64x64 results in a 2x2 final feature map.")
    print("   This is insufficient for classifying complex objects like Elephants or Guitars.")
    print("2. Digits (Row 2): 64x64 is actually larger than original (32x32).")
    print("   Information is fully preserved. 64x64 is perfect for Digits.")
    print("-" * 50)

if __name__ == '__main__':
    visualize_comparison()