import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from custom_dataset import LocalImageDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
import cv2
from cbam import CBAM
import argparse
import time
import random
from PIL import Image

# 导入字体设置工具
from font_utils import CHINESE_SUPPORTED, get_labels, suppress_font_warnings

# 抑制字体警告
suppress_font_warnings()

# 设置随机种子确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 在训练开始后启用benchmark以提升性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 初始化时禁用，训练时启用

def enable_cudnn_benchmark():
    """训练开始后启用cudnn benchmark以提升训练性能"""
    torch.backends.cudnn.benchmark = True
    print("已启用CUDNN benchmark以提升训练性能", flush=True)

set_seed(42)

print("=== VGG训练程序启动 ===", flush=True)
print(f"程序启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("正在导入模块...", flush=True)

class VGGRegressionCBAM(nn.Module):
    def __init__(self, freeze_features=True, debug_mode=False):
        super(VGGRegressionCBAM, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # 验证VGG16结构（调试模式）
        if debug_mode:
            print("VGG16特征层结构:", flush=True)
            for i, layer in enumerate(vgg.features):
                print(f"  Layer {i}: {layer}", flush=True)
        else:
            print(f"加载VGG16预训练模型，共{len(vgg.features)}层", flush=True)
        
        self.features = nn.Sequential()
        # VGG16的正确block分割：
        # Block1: 0-4 (Conv-ReLU-Conv-ReLU-MaxPool) -> 64 channels
        # Block2: 5-9 (Conv-ReLU-Conv-ReLU-MaxPool) -> 128 channels  
        # Block3: 10-16 (Conv-ReLU-Conv-ReLU-Conv-ReLU-MaxPool) -> 256 channels
        # Block4: 17-23 (Conv-ReLU-Conv-ReLU-Conv-ReLU-MaxPool) -> 512 channels
        # Block5: 24-30 (Conv-ReLU-Conv-ReLU-Conv-ReLU-MaxPool) -> 512 channels
        block_indices = [4, 9, 16, 23, 30]  # 每个block的最后一层索引
        in_channels = [64, 128, 256, 512, 512]
        last_idx = 0
        
        # 分别处理VGG层和CBAM层的冻结
        vgg_layers = []
        cbam_layers = []
        
        for i, idx in enumerate(block_indices):
            print(f"处理Block {i+1}: layers {last_idx}-{idx}, 输出通道数: {in_channels[i]}", flush=True)
            
            # 添加VGG层
            for j in range(last_idx, idx+1):
                layer_name = f'vgg_{j}'
                vgg_layer = vgg.features[j]
                self.features.add_module(layer_name, vgg_layer)
                vgg_layers.append(vgg_layer)
            
            # 添加CBAM层
            cbam_layer = CBAM(in_channels[i])
            cbam_name = f'cbam_{i+1}'
            self.features.add_module(cbam_name, cbam_layer)
            cbam_layers.append(cbam_layer)
            print(f"  添加CBAM层: {cbam_name}, 通道数: {in_channels[i]}", flush=True)
            
            last_idx = idx+1
        
        # 只冻结VGG的预训练层，保持CBAM层可训练
        if freeze_features:
            for layer in vgg_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            print("VGG预训练层已冻结，CBAM注意力层保持可训练状态", flush=True)
        
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
        
        # 权重初始化
        self._initialize_weights()
        
        # 验证模型结构
        self._verify_model_structure()
        
        # 打印可训练参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"模型参数统计:", flush=True)
        print(f"  总参数: {total_params:,}", flush=True)
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)", flush=True)
        print(f"  冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)", flush=True)
        
        # 计算模型大小
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        print(f"  模型大小: {model_size_mb:.1f} MB", flush=True)
        
    def _initialize_weights(self):
        """初始化CBAM层和回归头的权重"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # 只初始化CBAM中的卷积层，不影响预训练的VGG层
                if 'cbam' in name.lower():
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 只初始化CBAM中的BatchNorm层
                if 'cbam' in name.lower():
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        print("权重初始化完成: Xavier初始化线性层，Kaiming初始化CBAM卷积层", flush=True)
        
    def _verify_model_structure(self):
        """验证模型结构是否正确"""
        cbam_count = 0
        vgg_count = 0
        for name, module in self.features.named_modules():
            if 'cbam' in name and hasattr(module, 'channel_attention'):
                cbam_count += 1
            elif 'vgg' in name and isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
                vgg_count += 1
        
        print(f"模型结构验证:", flush=True)
        print(f"  检测到CBAM层数量: {cbam_count}", flush=True)
        print(f"  检测到VGG层数量: {vgg_count}", flush=True)
        
        if cbam_count != 5:
            print(f"警告: 期望5个CBAM层，实际检测到{cbam_count}个", flush=True)
        else:
            print("✓ CBAM层数量正确", flush=True)
        
    def forward(self, x):
        x = self.features(x)
        x = self.reg_head(x)
        return x

# Grad-CAM可视化工具
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        handle1 = self.target_layer.register_forward_hook(forward_hook)
        handle2 = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle1, handle2])
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    def __call__(self, input_tensor, target_index=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_index is None:
            target_index = 0
        loss = output[:, 0].sum() if output.ndim == 2 else output.sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        self.remove_hooks()
        return cam

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, start_epoch=0, use_amp=True, scheduler=None, model_save_path='best_vgg_model.pth'):
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10  # 早停耐心值
    
    # 重新启用混合精度训练
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    print(f"训练配置: 早停耐心值={early_stop_patience}, 梯度裁剪=1.0, 混合精度={use_amp}, 权重衰减=1e-4", flush=True)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\n开始Epoch {epoch + 1}/{num_epochs} - {time.strftime('%H:%M:%S')}", flush=True)
        print("正在加载第一个batch...", flush=True)
        
        batch_10_start_time = time.time()  # 每10个batch的计时
        
        for i, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            
            if i == 0:
                print(f"成功加载第一个batch，shape: {inputs.shape} - {time.strftime('%H:%M:%S')}", flush=True)
                print(f"实际batch_size: {inputs.shape[0]}", flush=True)
                print(f"输入数据类型: {inputs.dtype}, 设备: {inputs.device}", flush=True)
                
                # 显示GPU内存使用情况
                if torch.cuda.is_available():
                    print(f"训练开始时GPU内存: 已用={torch.cuda.memory_allocated() / 1024**3:.2f}GB, 缓存={torch.cuda.memory_reserved() / 1024**3:.2f}GB", flush=True)
            
            # 重新测量数据加载时间（从上一个batch结束到当前batch开始）
            if i > 0:
                data_load_time = batch_start_time - last_batch_end_time
            else:
                data_load_time = 0.0  # 第一个batch的加载时间已经包含在启动时间中
                
            # 只在前3个batch显示数据加载耗时
            if i < 3:
                print(f"Batch {i+1}: 数据加载耗时 {data_load_time:.2f}秒, batch_size={inputs.shape[0]}", flush=True)
            
            compute_start_time = time.time()
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)
            
            gpu_transfer_time = time.time() - compute_start_time
            if i == 0:
                print(f"数据已移至GPU，传输耗时 {gpu_transfer_time:.2f}秒 - {time.strftime('%H:%M:%S')}", flush=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            forward_start_time = time.time()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            torch.cuda.synchronize()  # 等待GPU计算完成
            forward_time = time.time() - forward_start_time
            
            backward_start_time = time.time()
            if use_amp and scaler:
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            torch.cuda.synchronize()  # 等待GPU计算完成
            backward_time = time.time() - backward_start_time
            
            total_batch_time = time.time() - batch_start_time
            last_batch_end_time = time.time()  # 记录当前batch结束时间
            
            # 只在前3个batch显示详细耗时
            if i < 3:
                print(f"Batch {i+1} 详细耗时: 数据加载={data_load_time:.2f}s, 前向传播={forward_time:.2f}s, 反向传播={backward_time:.2f}s, 总计={total_batch_time:.2f}s", flush=True)
            
            running_loss += loss.item()
            
            # 每10个batch的性能监测
            if (i + 1) % 10 == 0:
                batch_10_time = time.time() - batch_10_start_time
                avg_batch_time = batch_10_time / 10
                
                # GPU内存监控
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_cached = torch.cuda.memory_reserved() / 1024**3
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] 训练损失: {running_loss / 10:.3f}, 最近10batch耗时: {batch_10_time:.2f}s, 平均每batch: {avg_batch_time:.2f}s, GPU内存: {gpu_memory:.1f}GB/{gpu_cached:.1f}GB', flush=True)
                else:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] 训练损失: {running_loss / 10:.3f}, 最近10batch耗时: {batch_10_time:.2f}s, 平均每batch: {avg_batch_time:.2f}s', flush=True)
                
                running_loss = 0.0
                batch_10_start_time = time.time()  # 重置10batch计时器
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float().view(-1, 1)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        predictions = model(inputs)
                        loss = criterion(predictions, targets)
                else:
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        val_loss /= len(val_loader)
        r2 = r2_score(val_targets, val_predictions)
        
        # 学习率调度器更新
        if scheduler:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1} 验证损失: {val_loss:.3f}, R² 分数: {r2:.3f}, 当前学习率: {current_lr:.6f}')
        else:
            print(f'Epoch {epoch + 1} 验证损失: {val_loss:.3f}, R² 分数: {r2:.3f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f'保存最佳模型到 {model_save_path}，验证损失: {val_loss:.3f}')
        else:
            patience_counter += 1
            print(f'验证损失未改善，耐心计数器: {patience_counter}/{early_stop_patience}')
            
            if patience_counter >= early_stop_patience:
                print(f'早停触发！连续{early_stop_patience}个epoch验证损失未改善，停止训练。')
                break
        
        if epoch % 5 == 0:
            # 获取标签
            labels = get_labels(CHINESE_SUPPORTED)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(val_targets, val_predictions, alpha=0.5)
            plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
            plt.xlabel(labels['true_concentration'])
            plt.ylabel(labels['predicted_concentration'])
            plt.title(f"{labels['epoch']} {epoch + 1} {labels['prediction_vs_true']} (R² = {r2:.3f})")
            plt.savefig(f'vgg_prediction_plot_epoch_{epoch + 1}.png')
            plt.close()
    
    print(f"训练完成！最佳验证损失: {best_val_loss:.3f}", flush=True)

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU缓存", flush=True)

def main():
    print("=== 进入main函数 ===", flush=True)
    print("正在解析命令行参数...", flush=True)
    parser = argparse.ArgumentParser(description='VGG分组训练开关')
    parser.add_argument('--group', type=str, default='all', choices=['all', 'bg0', 'bg1', 'allgroup'],
                        help='选择训练数据分组: all(全部), bg0(不补光), bg1(补光), allgroup(全部分组)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='数据集路径，如果不指定则使用默认路径')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小，如果不指定则自动根据GPU显存调整')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数 (默认: 30)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='基础学习率 (默认: 0.001)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式，显示详细的模型结构信息')
    args = parser.parse_args()
    print(f"命令行参数解析完成: group={args.group}, data_path={args.data_path}, batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, debug={args.debug}", flush=True)

    print("正在检测CUDA设备...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)
    print(f"CUDA可用: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}", flush=True)
        print(f"当前GPU: {torch.cuda.get_device_name(0)}", flush=True)
    
    # 全局配置参数（避免变量作用域问题）
    batch_size = args.batch_size if args.batch_size else 192  # 使用命令行参数或默认值
    num_workers = 6   # 优化配置（快）
    use_amp = True    # 混合精度训练开关
    base_lr = args.lr  # 使用命令行参数
    num_epochs = args.epochs  # 使用命令行参数
    
    # 智能batch_size调整（仅在未指定时）
    if args.batch_size is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"检测到GPU显存: {gpu_memory_gb:.1f} GB", flush=True)
            
            if gpu_memory_gb >= 20:  # RTX 4090等高端卡
                batch_size = 192
                print("高端GPU配置: batch_size=192", flush=True)
            elif gpu_memory_gb >= 10:  # RTX 3080等中端卡
                batch_size = 96
                print("中端GPU配置: batch_size=96", flush=True)
            else:  # 低端GPU
                batch_size = 32
                print("低端GPU配置: batch_size=32", flush=True)
        else:
            batch_size = 16
            print("CPU模式: batch_size=16", flush=True)
    else:
        print(f"使用指定的batch_size: {batch_size}", flush=True)
    
    # 优化DataLoader配置
    pin_memory = True    # 优化配置（快）
    
    print(f"最终配置: batch_size={batch_size}, num_workers={num_workers}, use_amp={use_amp}, pin_memory={pin_memory}", flush=True)
    
    # GPU内存监控
    if torch.cuda.is_available():
        print(f"GPU内存状态: {torch.cuda.get_device_name(0)}")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"当前已用显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"当前缓存显存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 数据增强策略：训练和验证使用不同的transform
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # 科学实验图像的保守数据增强
        transforms.RandomRotation(degrees=5),  # 小角度旋转，模拟相机角度微调
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 减少颜色扰动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("数据增强策略:", flush=True)
    print("  训练集: 小角度旋转(±5°) + 轻微颜色扰动", flush=True)
    print("  验证/测试集: 仅标准化，无增强", flush=True)
    
    # 智能数据路径检测
    if args.data_path:
        dataset_path = args.data_path
    else:
        # 尝试多个可能的路径
        possible_paths = [
            r"/root/autodl-tmp/2025年实验照片",  # Linux云服务器路径
            r"D:\gm\2025年实验照片",  # Windows本地路径
            r".\2025年实验照片",  # 相对路径
            r"2025年实验照片"  # 当前目录
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                print(f"自动检测到数据路径: {dataset_path}", flush=True)
                break
        
        if dataset_path is None:
            print("错误：未找到数据集！请使用 --data_path 参数指定数据集路径", flush=True)
            print("可能的路径:", flush=True)
            for path in possible_paths:
                print(f"  {path}", flush=True)
            return
    
    # 添加路径检查和调试信息
    print(f"使用数据路径: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"错误：数据路径不存在！请检查路径: {dataset_path}")
        return
    
    print(f"数据路径存在，开始扫描图像文件...")

    if args.group in ['all', 'allgroup']:
        print("\n【全部数据训练】")
        print("正在初始化数据集，请稍候...")
        
        # 创建完整数据集用于划分
        full_dataset = LocalImageDataset(root_dir=dataset_path, transform=None)
        print(f"数据集初始化完成，共找到 {len(full_dataset)} 张图像")
        if len(full_dataset) == 0:
            print("错误：未找到任何有效的图像文件！")
            return
            
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        # 正确的数据集划分方式
        generator = torch.Generator().manual_seed(42)  # 确保可重复性
        train_dataset_base, val_dataset_base, test_dataset_base = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        
        # 创建自定义数据集类来应用不同的transform
        class TransformDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, transform, original_dataset):
                self.base_dataset = base_dataset
                self.transform = transform
                self.original_dataset = original_dataset
                
            def __len__(self):
                return len(self.base_dataset)
                
            def __getitem__(self, idx):
                # 获取原始索引
                original_idx = self.base_dataset.indices[idx]
                # 从原始数据集获取原始图像和标签
                img_path = self.original_dataset.image_files[original_idx]
                concentration = self.original_dataset.concentrations[original_idx]
                
                # 简化的图像加载（数据集已预先验证）
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception as e:
                    # 理论上不应该到达这里，但保留作为最后防线
                    print(f"意外错误：无法读取已验证的图像 {img_path}: {e}")
                    img = Image.new('RGB', (224, 224), color='black')
                
                if self.transform:
                    img = self.transform(img)
                    
                return img, concentration
        
        # 创建带有不同transform的数据集
        train_dataset = TransformDataset(train_dataset_base, train_transform, full_dataset)
        val_dataset = TransformDataset(val_dataset_base, val_test_transform, full_dataset)
        test_dataset = TransformDataset(test_dataset_base, val_test_transform, full_dataset)
        
        print(f"数据集划分:")
        print(f"训练集: {len(train_dataset)} 张图像 (使用数据增强)")
        print(f"验证集: {len(val_dataset)} 张图像 (无增强)")
        print(f"测试集: {len(test_dataset)} 张图像 (无增强)")
        
        print(f"RTX 4090高性能配置 - batch_size={batch_size}, num_workers={num_workers}", flush=True)
        print("大幅增加batch_size充分利用GPU，目标显存使用18-22GB", flush=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        model = VGGRegressionCBAM(freeze_features=True, debug_mode=args.debug).to(device)
        criterion = nn.MSELoss()
        
        # 根据batch_size调整学习率（大batch训练最佳实践）
        lr = base_lr * (batch_size / 32)  # 线性缩放
        
        # 优化器配置：训练所有可训练参数（包括CBAM层和回归头）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        
        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        print(f"学习率调整: base_lr={base_lr}, batch_size={batch_size}, final_lr={lr:.6f}", flush=True)
        print(f"优化器配置: 训练{len(trainable_params)}个可训练参数（包括CBAM层和回归头）", flush=True)
        
        start_epoch = 0
        best_model_path = 'best_vgg_model.pth'
        if os.path.exists(best_model_path):
            print('检测到已有best_vgg_model.pth，正在加载权重并断点续训...')
            try:
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                print("权重加载成功", flush=True)
            except Exception as e:
                print(f"权重加载失败: {e}，将从头开始训练", flush=True)
        
        # 启用CUDNN benchmark以提升性能
        enable_cudnn_benchmark()
        
        print("\n开始训练模型...")
        print("正在进入第一个epoch，数据加载可能需要几分钟时间...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, start_epoch=start_epoch, use_amp=use_amp, scheduler=scheduler, model_save_path=best_model_path)
        
        # 测试阶段
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        test_predictions = []
        test_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        predictions = model(inputs)
                else:
                    predictions = model(inputs)
                test_predictions.extend(predictions.cpu().numpy())
                test_targets.extend(targets.numpy())
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        mse = mean_squared_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)
        print("\n全部数据测试集结果:")
        print(f"均方误差 (MSE): {mse:.3f}")
        print(f"R² 分数: {r2:.3f}")
        
        # 获取标签
        labels = get_labels(CHINESE_SUPPORTED)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(test_targets, test_predictions, alpha=0.5)
        plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
        plt.xlabel(labels['true_concentration'])
        plt.ylabel(labels['predicted_concentration'])
        plt.title(f"{labels['all_data']}{labels['test_results']} (R² = {r2:.3f})")
        plt.savefig('vgg_final_prediction_plot_all.png')
        plt.close()
        
        # Grad-CAM可视化
        print("生成全部数据模型的Grad-CAM可视化...")
        target_layer = None
        for name, module in model.features.named_modules():
            if 'cbam_5' in name:
                target_layer = module
        assert target_layer is not None, '未找到目标CBAM层'
        grad_cam = GradCAM(model, target_layer)
        
        # 使用验证transform获取图像用于可视化
        vis_dataset = LocalImageDataset(root_dir=dataset_path, transform=val_test_transform)
        img, _ = vis_dataset[0]
        input_tensor = img.unsqueeze(0).to(device)
        cam = grad_cam(input_tensor)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        cv2.imwrite('vgg_grad_cam_all.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print('Grad-CAM可视化结果已保存为 vgg_grad_cam_all.png')

    if args.group in ['bg0', 'bg1', 'allgroup']:
        group_list = [args.group] if args.group in ['bg0', 'bg1'] else ['bg0', 'bg1']
        for bg_type in group_list:
            print(f"\n【仅{bg_type}图像训练】")
            
            # 创建数据集用于划分
            full_bg_dataset = LocalImageDataset(root_dir=dataset_path, transform=None, bg_type=bg_type)
            if len(full_bg_dataset) < 10:
                print(f"{bg_type}样本过少({len(full_bg_dataset)}张)，跳过...")
                continue
                
            train_size = int(0.7 * len(full_bg_dataset))
            val_size = int(0.15 * len(full_bg_dataset))
            test_size = len(full_bg_dataset) - train_size - val_size
            
            # 确保每个集合至少有1个样本
            if val_size == 0:
                val_size = 1
                train_size -= 1
            if test_size == 0:
                test_size = 1
                train_size -= 1
                
            # 正确的数据集划分方式
            generator = torch.Generator().manual_seed(42)  # 确保可重复性
            train_dataset_base, val_dataset_base, test_dataset_base = torch.utils.data.random_split(
                full_bg_dataset, [train_size, val_size, test_size], generator=generator
            )
            
            # 创建自定义数据集类来应用不同的transform
            class TransformDatasetBG(torch.utils.data.Dataset):
                def __init__(self, base_dataset, transform, original_dataset):
                    self.base_dataset = base_dataset
                    self.transform = transform
                    self.original_dataset = original_dataset
                    
                def __len__(self):
                    return len(self.base_dataset)
                    
                def __getitem__(self, idx):
                    # 获取原始索引
                    original_idx = self.base_dataset.indices[idx]
                    # 从原始数据集获取原始图像和标签
                    img_path = self.original_dataset.image_files[original_idx]
                    concentration = self.original_dataset.concentrations[original_idx]
                    
                    # 简化的图像加载（数据集已预先验证）
                    try:
                        img = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        # 理论上不应该到达这里，但保留作为最后防线
                        print(f"意外错误：无法读取已验证的图像 {img_path}: {e}")
                        img = Image.new('RGB', (224, 224), color='black')
                    
                    if self.transform:
                        img = self.transform(img)
                        
                    return img, concentration
            
            # 创建带有不同transform的数据集
            train_dataset = TransformDatasetBG(train_dataset_base, train_transform, full_bg_dataset)
            val_dataset = TransformDatasetBG(val_dataset_base, val_test_transform, full_bg_dataset)
            test_dataset = TransformDatasetBG(test_dataset_base, val_test_transform, full_bg_dataset)
            
            print(f"{bg_type}数据集划分:")
            print(f"训练集: {len(train_dataset)} 张图像 (使用数据增强)")
            print(f"验证集: {len(val_dataset)} 张图像 (无增强)")
            print(f"测试集: {len(test_dataset)} 张图像 (无增强)")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            model = VGGRegressionCBAM(freeze_features=True, debug_mode=args.debug).to(device)
            criterion = nn.MSELoss()
            
            # 根据batch_size调整学习率（大batch训练最佳实践）
            lr = base_lr * (batch_size / 32)  # 线性缩放
            
            # 优化器配置：训练所有可训练参数（包括CBAM层和回归头）
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
            
            # 添加学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            print(f"学习率调整: base_lr={base_lr}, batch_size={batch_size}, final_lr={lr:.6f}", flush=True)
            print(f"优化器配置: 训练{len(trainable_params)}个可训练参数（包括CBAM层和回归头）", flush=True)
            
            best_model_path = f'best_vgg_model_{bg_type}.pth'
            start_epoch = 0
            
            # 断点续训逻辑
            if os.path.exists(best_model_path):
                print(f'检测到已有{best_model_path}，正在加载权重并断点续训...')
                try:
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                    print("权重加载成功", flush=True)
                except Exception as e:
                    print(f"权重加载失败: {e}，将从头开始训练", flush=True)
            
            # 启用CUDNN benchmark以提升性能
            enable_cudnn_benchmark()
            
            print(f"\n开始训练{bg_type}模型...")
            train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs, start_epoch=start_epoch, use_amp=use_amp, scheduler=scheduler, model_save_path=best_model_path)
            
            # 测试阶段
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            test_predictions = []
            test_targets = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            predictions = model(inputs)
                    else:
                        predictions = model(inputs)
                    test_predictions.extend(predictions.cpu().numpy())
                    test_targets.extend(targets.numpy())
            test_predictions = np.array(test_predictions)
            test_targets = np.array(test_targets)
            mse = mean_squared_error(test_targets, test_predictions)
            r2 = r2_score(test_targets, test_predictions)
            print(f"\n{bg_type}测试集结果:")
            print(f"均方误差 (MSE): {mse:.3f}")
            print(f"R² 分数: {r2:.3f}")
            
            # 获取标签
            labels = get_labels(CHINESE_SUPPORTED)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(test_targets, test_predictions, alpha=0.5)
            plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
            plt.xlabel(labels['true_concentration'])
            plt.ylabel(labels['predicted_concentration'])
            plt.title(f'{bg_type}{labels["test_results"]} (R² = {r2:.3f})')
            plt.savefig(f'vgg_final_prediction_plot_{bg_type}.png')
            plt.close()
            
            # Grad-CAM可视化
            print(f"生成{bg_type}模型的Grad-CAM可视化...")
            target_layer = None
            for name, module in model.features.named_modules():
                if 'cbam_5' in name:
                    target_layer = module
            assert target_layer is not None, '未找到目标CBAM层'
            grad_cam = GradCAM(model, target_layer)
            
            # 使用验证transform获取图像用于可视化
            vis_dataset = LocalImageDataset(root_dir=dataset_path, transform=val_test_transform, bg_type=bg_type)
            img, _ = vis_dataset[0]
            input_tensor = img.unsqueeze(0).to(device)
            cam = grad_cam(input_tensor)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cm.jet(cam_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
            cv2.imwrite(f'vgg_grad_cam_{bg_type}.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f'Grad-CAM可视化结果已保存为 vgg_grad_cam_{bg_type}.png')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断", flush=True)
    except Exception as e:
        print(f"\n程序运行出错: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("程序结束，已清理GPU缓存", flush=True) 