import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader

# 将父目录加入路径以便导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入底层数据处理模块
from utils.ntl_utils.getdata import Cus_Dataset
import utils.ntl_utils.utils_digit as utils_digit
import utils.ntl_utils.utils_pacs as utils_pacs
# 使用官方工具加载模型架构
from utils.load_utils import load_model as official_load_model

# ==========================================
# 用户配置区
# ==========================================

DOMAIN_SEQUENCES = {
    'digits': ['rmt0', 'rmt15', 'rmt30', 'rmt45', 'rmt60', 'rmt75'],
    'pacs': ['pacs_p', 'pacs_a', 'pacs_c', 'pacs_s']
}

MODEL_PATHS = {
    'digits': {
        'SL': './saved_models/SL_rmt0_vgg13.pth',
        'tNTL': './saved_models/tNTL_rmt0_rmt45_vgg13.pth',
        'HNTL': './saved_models/tHNTL_rmt0_rmt45_vgg13.pth',
        'CUTI': './saved_models/tCUTI_rmt0_rmt45_vgg13.pth',
        'SOPHON': './saved_models/tSOPHON_rmt0_rmt45_vgg13.pth',
    },
    'pacs': {
        'SL': './saved_models/SL_pacs_p_vgg13.pth',
        'tNTL': './saved_models/tNTL_pacs_p_pacs_s_vgg13.pth',
        'HNTL': './saved_models/tHNTL_pacs_p_pacs_s_vgg13.pth',
        'CUTI': './saved_models/tCUTI_pacs_p_pacs_s_vgg13.pth',
        'SOPHON': './saved_models/tSOPHON_pacs_p_pacs_s_vgg13.pth',
    }
}

# ==========================================
# 配置类 (对齐您的训练设置)
# ==========================================

class MockConfig:
    def __init__(self, args, method_name=None):
        self.dataset = args.dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # [关键修正] 统一使用 128x128 分辨率 (PACS必须是128)
        self.image_size = 128 if args.dataset == 'pacs' else 64
        
        if args.dataset == 'pacs':
            self.teacher_network = 'vgg13' 
            self.num_classes = 7
            self.teacher_pretrain = True 
        else: # digits
            self.teacher_network = 'vgg13'
            self.num_classes = 10
            self.teacher_pretrain = False
            
        self.task_name = method_name if method_name else 'SL'
        self.domain_src = 'pacs_p' if args.dataset == 'pacs' else 'rmt0' 

# ==========================================
# 数据加载
# ==========================================

def get_raw_data_loader(domain_name):
    # Digits
    if domain_name == 'rmt0': return utils_digit.get_mnist_rotate_0()
    if domain_name == 'rmt15': return utils_digit.get_mnist_rotate_15()
    if domain_name == 'rmt30': return utils_digit.get_mnist_rotate_30()
    if domain_name == 'rmt45': return utils_digit.get_mnist_rotate_45()
    if domain_name == 'rmt60': return utils_digit.get_mnist_rotate_60()
    if domain_name == 'rmt75': return utils_digit.get_mnist_rotate_75()
    # PACS
    if domain_name == 'pacs_p': return utils_pacs.get_pacs_P()
    if domain_name == 'pacs_a': return utils_pacs.get_pacs_A()
    if domain_name == 'pacs_c': return utils_pacs.get_pacs_C()
    if domain_name == 'pacs_s': return utils_pacs.get_pacs_S()
    
    raise ValueError(f"Unknown domain: {domain_name}")

def get_dataloader(domain_name, config, batch_size=64):
    raw_data = get_raw_data_loader(domain_name)
    dataset = Cus_Dataset(
        mode='val', 
        dataset_1=raw_data, 
        begin_ind1=0, 
        size1=raw_data[2],
        config=config
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader

# ==========================================
# 模型加载
# ==========================================

def load_method_model(method_name, path, config):
    if not os.path.exists(path):
        print(f"Warning: Model path not found: {path}")
        return None
        
    print(f"Loading {method_name} ({config.teacher_network}) from {path}...")
    
    try:
        model = official_load_model(config)
    except Exception as e:
        print(f"Error building model: {e}")
        return None
    
    try:
        checkpoint = torch.load(path, map_location=config.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()
            
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        
    except Exception as e:
        print(f"  [Error] Failed to load weights: {e}")
        return None
        
    model.eval()
    return model

# ==========================================
# 辅助函数：标签形状修正
# ==========================================
def fix_label_shape(labels):
    """
    统一处理各种奇形怪状的 Label，目标是输出 (Batch_Size,) 的 LongTensor
    输入可能是: (B, 1, C), (B, C), (B, 1), (B,)
    """
    # 1. 如果有中间的 singleton 维度 (B, 1, ...)，先压缩
    if labels.dim() > 1 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
        
    # 2. 检查是否为 One-Hot (B, C) 其中 C > 1
    # 我们检查最后一维，如果 > 1 则认为是类别概率/One-Hot
    if labels.dim() > 1 and labels.shape[-1] > 1:
        labels = torch.argmax(labels, dim=-1)
        
    # 3. 经过上述处理，如果还有多余维度 (B, 1)，再次展平为 (B,)
    if labels.dim() > 1:
        labels = labels.view(-1)
        
    return labels

# ==========================================
# 评估与特征提取
# ==========================================

def extract_features(model, dataloader, device, max_samples=500):
    features_list = []
    targets_list = []
    model.eval()
    count = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # --- 修复标签形状 ---
            labels = fix_label_shape(labels)
            # -------------------
            
            # VGG 特征提取
            if hasattr(model, 'forward_f'): 
                _, feature = model.forward_f(imgs)
            elif hasattr(model, 'features'):
                feature = model.features(imgs)
                feature = feature.view(feature.size(0), -1)
            else:
                feature = model(imgs) 

            features_list.append(feature.cpu().numpy())
            targets_list.append(labels.cpu().numpy())
            count += imgs.size(0)
            if count >= max_samples: break
            
    return np.concatenate(features_list)[:max_samples], np.concatenate(targets_list)[:max_samples]

def evaluate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # --- 修复标签形状 ---
            labels = fix_label_shape(labels)
            # -------------------

            outputs = model(imgs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            if isinstance(outputs, dict): outputs = outputs['pred']
            
            _, predicted = torch.max(outputs.data, 1)
            
            # 确保 predicted 和 labels 形状一致再比较
            if predicted.shape != labels.shape:
                print(f"Shape mismatch! Pred: {predicted.shape}, Label: {labels.shape}")
                # 尝试强制对齐
                labels = labels.view_as(predicted)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ==========================================
# 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='digits', choices=['digits', 'pacs'])
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    base_config = MockConfig(args)
    paths = MODEL_PATHS[args.dataset]
    domains = DOMAIN_SEQUENCES[args.dataset]
    
    results = {m: [] for m in paths.keys()}
    feature_storage = {m: {} for m in paths.keys()}
    
    # 1. 加载模型
    models_loaded = {}
    print(f"\n[Init] Evaluation: {args.dataset}, Size: {base_config.image_size}, Net: {base_config.teacher_network}")
    for method, path in paths.items():
        # 更新 MockConfig 以匹配当前方法的 task_name
        method_config = MockConfig(args, method_name=method)
        model = load_method_model(method, path, method_config)
        if model: models_loaded[method] = model

    if not models_loaded:
        print("No models loaded. Please check MODEL_PATHS.")
        return

    # 2. 评估
    for domain in domains:
        print(f"\n>>> Domain: {domain}")
        try:
            loader = get_dataloader(domain, base_config)
        except Exception as e:
            print(f"Skipping {domain}: {e}")
            continue
        
        for method, model in models_loaded.items():
            acc = evaluate_accuracy(model, loader, base_config.device)
            results[method].append(acc)
            print(f"  {method}: {acc:.2f}%")
            
            feat, _ = extract_features(model, loader, base_config.device, max_samples=300)
            feature_storage[method][domain] = feat

    # 3. 绘图 - 曲线
    plt.figure(figsize=(10, 6))
    styles = {'SL': 'k--', 'tNTL': 'b^-', 'HNTL': 'gs-', 'CUTI': 'D-', 'SOPHON': 'm*-'}
    for method, accs in results.items():
        if not accs: continue
        fmt = styles.get(method, 'o-')
        plt.plot(range(len(accs)), accs, fmt, label=method)
    
    plt.xticks(range(len(domains)), domains, rotation=45)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Performance Barrier Curve ({args.dataset})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs('third_party/results', exist_ok=True)
    plt.savefig(f'third_party/results/curve_{args.dataset}.png')
    
    # 4. 绘图 - t-SNE
    print("\nGenerating t-SNE...")
    for method, domain_data in feature_storage.items():
        if not domain_data: continue
        X_list, y_list = [], []
        for i, d in enumerate(domains):
            if d in domain_data:
                X_list.append(domain_data[d])
                y_list.extend([i]*len(domain_data[d]))
        
        if not X_list: continue
        X = np.concatenate(X_list)
        X += 1e-4 * np.random.randn(*X.shape)
        
        tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), init='pca', learning_rate='auto')
        try:
            X_emb = tsne.fit_transform(X)
        except: continue
        
        plt.figure(figsize=(8,8))
        colors = sns.color_palette("hsv", len(domains))
        for i, d in enumerate(domains):
            idx = (np.array(y_list) == i)
            if idx.any():
                plt.scatter(X_emb[idx,0], X_emb[idx,1], label=d, s=20, alpha=0.6, color=colors[i])
        
        plt.legend()
        plt.axis('off')
        plt.savefig(f'third_party/results/tsne_{args.dataset}_{method}.png')
        plt.close()
    
    print("\nDone! Results saved in third_party/results/")

if __name__ == '__main__':
    main()