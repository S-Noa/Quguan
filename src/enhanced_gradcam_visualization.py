#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版Grad-CAM可视化工具（修复版）
支持多种数据集选择：干净数据集 vs 特征数据集
支持自主选择模型路径
处理不同像素尺寸的图像
"""

import torch
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# PyTorch 2.6+ 兼容性修复（避免递归）
def apply_pytorch_compatibility():
    """应用PyTorch兼容性修复"""
    # 在模块级别保存原始函数
    import types
    
    # 获取真正的原始torch.load
    original_torch_load = getattr(torch, '_original_load', None)
    if original_torch_load is None:
        # 第一次运行，保存原始函数
        original_torch_load = torch.load
        torch._original_load = original_torch_load
    
    # 添加安全全局变量
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            torch.serialization.add_safe_globals([argparse.Namespace])
        except:
            pass
    
    def safe_torch_load(path, map_location=None, **kwargs):
        """安全的torch.load函数"""
        # 清理参数
        clean_kwargs = {k: v for k, v in kwargs.items() if k != 'weights_only'}
        
        try:
            # 尝试PyTorch 2.6+
            return original_torch_load(path, map_location=map_location, weights_only=False, **clean_kwargs)
        except TypeError as e:
            if "unexpected keyword argument 'weights_only'" in str(e):
                # 回退到旧版本
                return original_torch_load(path, map_location=map_location, **clean_kwargs)
            raise
        except Exception as e:
            if "argparse.Namespace" in str(e) or "Weights only load failed" in str(e):
                # 处理Namespace错误
                try:
                    if hasattr(torch.serialization, 'safe_globals'):
                        with torch.serialization.safe_globals([argparse.Namespace]):
                            return original_torch_load(path, map_location=map_location, weights_only=True, **clean_kwargs)
                except:
                    pass
                # 最终回退
                return original_torch_load(path, map_location=map_location, **clean_kwargs)
            raise
    
    # 只在需要时替换
    if not hasattr(torch.load, '_is_safe_load'):
        torch.load = safe_torch_load
        torch.load._is_safe_load = True
    
    return True

# 应用修复
apply_pytorch_compatibility()
import torch.nn as nn
from torchvision import transforms
from cnn_model import CNNFeatureExtractor
from advanced_cnn_models import AdvancedLaserSpotCNN
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import os
import glob
import json
import argparse
from PIL import Image
from pathlib import Path

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib的中文字体"""
    try:
        chinese_fonts = [
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',  # 宋体
            'KaiTi',   # 楷体
            'FangSong'  # 仿宋
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用中文字体: {font}")
                return font
        
        print("❌ 未找到中文字体，将使用英文标签")
        return None
        
    except Exception as e:
        print(f"字体设置失败: {e}")
        return None

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
    
    def __call__(self, input_tensor, model_type='traditional_cnn'):
        # 使用无hook的方法避免视图修改错误
        return self._gradcam_without_hooks(input_tensor, model_type)
    
    def _gradcam_without_hooks(self, input_tensor, model_type):
        """直接使用传统hook方法，之前的无hook方法有梯度问题"""
        print("🔄 直接使用传统hook方法...")
        return self._fallback_gradcam(input_tensor, model_type)
    
    def _fallback_gradcam(self, input_tensor, model_type):
        """增强版Grad-CAM修复方法 - 解决蓝色图像问题"""
        print("🔧 使用增强版Grad-CAM修复方法...")
        
        # 重置输入
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # 完全使用eval模式确保预测准确
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        # 🎯 关键修复1: 选择更好的目标层
        if model_type == 'resnet50' and hasattr(self.model, 'backbone'):
            # 尝试使用layer3而不是layer4（更早的层，更好的空间分辨率）
            layer3 = self.model.backbone.layer3
            if hasattr(layer3, '__getitem__') and len(layer3) > 0:
                last_block_3 = layer3[-1]
                if hasattr(last_block_3, 'conv2'):
                    target_layer = last_block_3.conv2
                    print(f"   🎯 使用layer3[-1].conv2作为目标层（更好的空间分辨率）")
                else:
                    target_layer = last_block_3
                    print(f"   🎯 使用layer3[-1]作为目标层")
            else:
                target_layer = self.target_layer
                print(f"   🎯 使用原始目标层")
        else:
            target_layer = self.target_layer
            print(f"   🎯 使用原始目标层")
        
        # 确保目标层参数需要梯度
        for param in target_layer.parameters():
            param.requires_grad_(True)
        
        print(f"   📊 模型模式: {'训练' if self.model.training else '评估'} (梯度已启用)")
        print(f"   📊 目标层参数需要梯度: {next(target_layer.parameters()).requires_grad}")
        
        # Hook变量
        activations = None
        gradients = None
        
        def save_activation(module, input, output):
            nonlocal activations
            activations = output
            print(f"   ✓ 激活捕获成功: {output.shape}, requires_grad: {output.requires_grad}")
        
        def save_gradient(module, grad_input, grad_output):
            nonlocal gradients
            if grad_output[0] is not None:
                gradients = grad_output[0]
                print(f"   ✓ 梯度捕获成功: {grad_output[0].shape}")
            else:
                print("   ❌ 梯度为None")
        
        # 注册hooks
        h1 = target_layer.register_forward_hook(save_activation)
        h2 = target_layer.register_backward_hook(save_gradient)
        
        try:
            # 前向传播（eval模式，预测准确）
            if model_type in ['resnet50', 'vgg']:
                output = self.model(input_tensor)
            elif model_type.startswith('enhanced_laser_spot_cnn'):
                output, _ = self.model(input_tensor)
            else:
                output, _ = self.model(input_tensor)
            
            print(f"   ✅ 前向传播成功: 输出形状 {output.shape}")
            print(f"   ✅ eval模式准确预测: {output.item():.6f}")
            
            # 🎯 关键修复2: 使用平方输出增强梯度
            target = (output ** 2).sum()
            print(f"   🔧 使用平方输出增强梯度: {target.item():.6f}")
            
            # 清零梯度
            self.model.zero_grad()
            
            # 反向传播
            target.backward(retain_graph=False)
            print("   ✓ 反向传播完成")
            
            # 移除hooks
            h1.remove()
            h2.remove()
            
            # 检查是否获取到激活和梯度
            if activations is None:
                print("   ❌ 未获取到激活值")
                return None
                
            if gradients is None:
                print("   ❌ 未获取到梯度")
                return None
            
            print(f"   ✓ 激活形状: {activations.shape}")
            print(f"   ✓ 梯度形状: {gradients.shape}")
            print(f"   ✓ 梯度统计: min={gradients.min():.6f}, max={gradients.max():.6f}, mean={gradients.mean():.6f}")
            
            # 🎯 关键修复3: 增强权重计算
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            # 放大权重以增强CAM对比度
            weights = weights * 3.0
            print(f"   ✓ 增强权重形状: {weights.shape}, 放大系数: 3.0")
            
            # 加权求和得到CAM
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            
            # 应用ReLU
            cam = torch.relu(cam)
            print(f"   ✓ CAM形状: {cam.shape}")
            
            # 🎯 关键修复4: 改进归一化和对比度增强
            cam_min = torch.min(cam)
            cam_max = torch.max(cam)
            
            if cam_max > cam_min:
                cam_normalized = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                # 伽马校正增强对比度
                cam_normalized = torch.pow(cam_normalized, 0.6)
                print(f"   ✓ 应用伽马校正 (γ=0.6) 增强对比度")
            else:
                print("   ⚠️ CAM值为常数，尝试使用原始激活值")
                # 如果CAM值为常数，尝试使用原始激活值
                cam_normalized = torch.mean(activations, dim=1, keepdim=True)
                cam_normalized = torch.relu(cam_normalized)
                cam_min = torch.min(cam_normalized)
                cam_max = torch.max(cam_normalized)
                if cam_max > cam_min:
                    cam_normalized = (cam_normalized - cam_min) / (cam_max - cam_min + 1e-8)
                else:
                    cam_normalized = torch.ones_like(cam_normalized) * 0.5  # 均匀分布
            
            print(f"   ✓ 修复后CAM范围: {cam_normalized.min():.6f} - {cam_normalized.max():.6f}")
            
            # 转换为numpy
            result = cam_normalized.squeeze().detach().cpu().numpy()
            print(f"   🎉 增强版GradCAM修复成功! 结果形状: {result.shape}")
            
            return result
            
        except Exception as e:
            # 确保hooks被移除
            try:
                h1.remove()
                h2.remove()
            except:
                pass
            print(f"   ❌ GradCAM计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def remove_hooks(self):
        # 兼容性方法，无hook版本不需要清理
        pass

def detect_model_type(model_path):
    """检测模型类型 - 支持增强激光光斑CNN、ResNet50、VGG、增强CNN等"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 处理检查点格式：可能是直接的state_dict，也可能包装在checkpoint中
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("🔍 检测到检查点格式，提取model_state_dict")
    else:
        state_dict = checkpoint
        print("🔍 检测到直接state_dict格式")
    
    # 获取所有层名用于调试
    layer_names = list(state_dict.keys())
    print(f"🔍 模型层名示例: {layer_names[:5]}...")
    
    # 检查增强激光光斑CNN（train_enhanced_cnn_with_6modes.py训练的模型）
    # 更强的检测逻辑：检查关键组件
    enhanced_cnn_indicators = [
        'enhanced_cbam',
        'multi_scale_fusion', 
        'fusion_conv',
        'laser_constraint',
        'regressor.0.weight'  # 增强CNN的回归头
    ]
    
    if any(any(indicator in name for name in layer_names) for indicator in enhanced_cnn_indicators):
        # 进一步检查骨干网络类型
        if any('backbone.features' in name for name in layer_names):
            print("✅ 检测到增强激光光斑CNN模型 (VGG骨干)")
            return 'enhanced_laser_spot_cnn_vgg'
        elif any('backbone.layer' in name for name in layer_names):
            print("✅ 检测到增强激光光斑CNN模型 (ResNet50骨干)")
            return 'enhanced_laser_spot_cnn_resnet'
        else:
            print("✅ 检测到增强激光光斑CNN模型 (VGG骨干-默认)")
            return 'enhanced_laser_spot_cnn_vgg'  # 默认为VGG
    
    # 检查ResNet50回归模型
    elif any('backbone.layer4' in name or 'regressor.0.weight' in name for name in layer_names):
        print("✅ 检测到ResNet50回归模型")
        return 'resnet50'
    elif 'backbone.conv1.weight' in state_dict and 'regressor' in str(layer_names):
        print("✅ 检测到ResNet50回归模型")
        return 'resnet50'
    
    # 检查VGG回归模型
    elif any('features.0.weight' in name or 'reg_head' in name or 'cbam' in name for name in layer_names):
        print("✅ 检测到VGG回归模型")
        return 'vgg'
    elif 'features.0.weight' in state_dict and ('reg_head.0.weight' in state_dict or 'classifier.0.weight' in state_dict):
        print("✅ 检测到VGG回归模型")
        return 'vgg'
    
    # 检查传统增强CNN模型（advanced_cnn_models.py）
    elif 'layer1.0.conv1.weight' in state_dict or 'classifier.0.weight' in state_dict:
        print("✅ 检测到传统增强CNN模型")
        return 'advanced_cnn'
        
    # 检查传统CNN模型
    elif 'conv1.weight' in state_dict and 'fc2.weight' in state_dict:
        print("✅ 检测到传统CNN模型")
        return 'traditional_cnn'
    
    else:
        print("⚠️  无法确定模型类型，默认使用传统CNN")
        print(f"🔍 请检查这些层名: {layer_names[:10]}")
        return 'traditional_cnn'

def load_model_safe(model_path, device):
    """安全加载模型，自动检测模型类型 - 支持ResNet50、VGG、增强CNN等"""
    print(f"🔍 检测模型: {model_path}")
    
    # 检测模型类型
    model_type = detect_model_type(model_path)
    print(f"📋 模型类型: {model_type}")
    
    # 根据类型创建相应的模型
    if model_type.startswith('enhanced_laser_spot_cnn'):
        print("🚀 创建增强激光光斑CNN模型...")
        # 导入增强激光光斑CNN模型
        try:
            sys.path.append('src')
            from enhanced_laser_spot_cnn import create_enhanced_laser_spot_model
            
            # 根据检测到的骨干网络类型选择
            if model_type == 'enhanced_laser_spot_cnn_vgg':
                backbone = 'vgg16'
            elif model_type == 'enhanced_laser_spot_cnn_resnet':
                backbone = 'resnet50'
            else:
                backbone = 'vgg16'  # 默认
            
            model = create_enhanced_laser_spot_model(
                backbone=backbone,
                constraint_weight=0.1,
                freeze_backbone=True
            ).to(device)
            
            print(f"   骨干网络: {backbone}")
            
        except ImportError as e:
            print(f"⚠️ 无法导入增强激光光斑CNN模型: {e}")
            print("   使用传统CNN作为备选...")
            model = CNNFeatureExtractor().to(device)
            model_type = 'traditional_cnn'
            
    elif model_type == 'resnet50':
        print("🚀 创建ResNet50回归模型...")
        # 导入ResNet50模型
        try:
            sys.path.append('cloud_vgg_training_package/src')
            from resnet_regression import ResNet50Regression
            model = ResNet50Regression(freeze_backbone=False).to(device)
        except ImportError:
            print("⚠️ 无法导入ResNet50模型，尝试从torchvision创建...")
            from torchvision import models
            import torch.nn as nn
            backbone = models.resnet50(weights='IMAGENET1K_V2')
            backbone.fc = nn.Identity()
            model = nn.Sequential(
                backbone,
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(2048, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.15),
                    nn.Linear(128, 1)
                )
            ).to(device)
            
    elif model_type == 'vgg':
        print("🚀 创建VGG回归模型...")
        # 导入VGG模型
        try:
            sys.path.append('cloud_vgg_training_package/src')
            from vgg_regression import VGGRegressionCBAM
            model = VGGRegressionCBAM(freeze_features=True).to(device)
        except ImportError:
            print("⚠️ 无法导入VGG模型，尝试从src目录创建...")
            try:
                sys.path.append('src')
                # 这里可以添加其他VGG模型的导入逻辑
                from vgg_regression import VGGRegressionCBAM
                model = VGGRegressionCBAM(freeze_features=True).to(device)
            except ImportError:
                print("❌ 无法创建VGG模型，使用传统CNN")
                model = CNNFeatureExtractor().to(device)
                model_type = 'traditional_cnn'
            
    elif model_type == 'advanced_cnn':
        print("🚀 创建高级CNN模型...")
        model = AdvancedLaserSpotCNN(
            num_features=512,
            use_attention=True,
            use_multiscale=True,
            use_laser_attention=False  # 特征数据集上禁用激光光斑专用注意力
        ).to(device)
    else:
        print("🔧 创建传统CNN模型...")
        model = CNNFeatureExtractor().to(device)
    
    # 加载权重 - 改进的加载逻辑
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理检查点格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("📂 从检查点中提取model_state_dict")
    else:
        state_dict = checkpoint
        print("📂 直接使用state_dict")
    
    # 对于增强激光光斑CNN，使用更宽松的加载策略
    if model_type.startswith('enhanced_laser_spot_cnn'):
        try:
            # 首先尝试直接加载（strict=False允许部分匹配）
            model.load_state_dict(state_dict, strict=False)
            print("✅ 增强CNN模型权重加载成功（strict=False）")
            
            # 计算匹配率
            model_dict = model.state_dict()
            matched_keys = [k for k in state_dict.keys() if k in model_dict and 
                          model_dict[k].shape == state_dict[k].shape]
            match_rate = len(matched_keys) / len(state_dict) * 100
            
        except Exception as e:
            print(f"⚠️ 直接加载失败: {e}")
            print("🔄 尝试过滤加载...")
            
            # 备选方案：过滤加载
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            match_rate = len(filtered_dict) / len(state_dict) * 100
            
    else:
        # 其他模型类型使用原来的过滤加载方式
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() 
                        if k in model_dict and model_dict[k].shape == v.shape}
        
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        match_rate = len(filtered_dict) / len(state_dict) * 100
    
    # 不设置model.eval()，让GradCAM自己控制模型模式
    
    print(f"✅ 模型加载完成: 权重匹配率 {match_rate:.1f}%")
    
    if match_rate < 50:
        print("⚠️  警告: 权重匹配率较低，可能影响模型性能")
    elif match_rate > 90:
        print("🎉 权重匹配率很高，模型应该工作正常")
    
    return model, model_type

def detect_available_models():
    """检测可用的已训练模型"""
    models = {}
    model_paths = {
        'bg0': [
            'best_model_bg0.pth',
            'baseline_cnn_bg0_results_20250604_174456/best_model.pth',
            'baseline_cnn_checkpoint_resume/best_model.pth',
            'models/best_model_bg0.pth'
        ],
        'bg1': [
            'best_model_bg1.pth',
            'models/best_model_bg1.pth'
        ],
        'all': [
            'best_model.pth',
            'models/best_model.pth'
        ]
    }
    
    for model_type, paths in model_paths.items():
        for path in paths:
            if os.path.exists(path):
                models[model_type] = path
                print(f"✅ 发现{model_type}模型: {path}")
                break
    
    return models

def find_feature_dataset_images(feature_dataset_path=None, bg_mode=None, power_filter=None):
    """查找特征数据集中的图像，支持6档细分模式过滤"""
    if feature_dataset_path is None:
        # 自动检测特征数据集 - 增强云端兼容性
        print("🔍 自动检测特征数据集...")
        
        # 候选目录列表（按优先级排序）
        candidate_dirs = []
        
        # 1. 带时间戳的特征数据集（本地常见格式）
        timestamped_datasets = glob.glob("feature_dataset_*")
        candidate_dirs.extend(timestamped_datasets)
        
        # 2. 简单的feature_dataset目录（云端常见格式）
        if os.path.exists("feature_dataset") and os.path.isdir("feature_dataset"):
            candidate_dirs.append("feature_dataset")
            
        # 3. 其他可能的变体
        for variant in ["Feature_Dataset", "FEATURE_DATASET", "feature-dataset"]:
            if os.path.exists(variant) and os.path.isdir(variant):
                candidate_dirs.append(variant)
        
        print(f"   找到候选目录: {candidate_dirs}")
        
        if not candidate_dirs:
            print("❌ 未找到特征数据集目录")
            print("   支持的目录格式: feature_dataset, feature_dataset_*, Feature_Dataset, feature-dataset")
            return []
        
        # 选择最佳候选目录（优先选择有效的目录）
        valid_datasets = []
        for dataset_dir in candidate_dirs:
            images_dir = os.path.join(dataset_dir, "images")
            info_dir = os.path.join(dataset_dir, "original_info")
            
            if os.path.exists(images_dir) and os.path.exists(info_dir):
                # 检查是否有实际内容
                image_count = len(glob.glob(os.path.join(images_dir, "*.jpg"))) + \
                             len(glob.glob(os.path.join(images_dir, "*.png"))) + \
                             len(glob.glob(os.path.join(images_dir, "*.jpeg")))
                info_count = len(glob.glob(os.path.join(info_dir, "*.json")))
                
                if image_count > 0 and info_count > 0:
                    valid_datasets.append({
                        'path': dataset_dir,
                        'image_count': image_count,
                        'info_count': info_count,
                        'mtime': os.path.getmtime(dataset_dir)
                    })
                    print(f"   ✅ 有效数据集: {dataset_dir} (图像: {image_count}, 信息: {info_count})")
                else:
                    print(f"   ⚠️ 空数据集: {dataset_dir} (图像: {image_count}, 信息: {info_count})")
            else:
                print(f"   ❌ 无效结构: {dataset_dir}")
        
        if not valid_datasets:
            print("❌ 未找到有效的特征数据集")
            return []
        
        # 选择最新的有效数据集
        feature_dataset_path = max(valid_datasets, key=lambda x: x['mtime'])['path']
        print(f"🎯 选择数据集: {feature_dataset_path}")
    
    # 验证选定的数据集结构
    
    images_dir = os.path.join(feature_dataset_path, "images")
    info_dir = os.path.join(feature_dataset_path, "original_info")
    
    if not os.path.exists(images_dir):
        print(f"❌ 特征数据集图像目录不存在: {images_dir}")
        return []
    
    if not os.path.exists(info_dir):
        print(f"❌ 特征数据集信息目录不存在: {info_dir}")
        return []
    
    print(f"🔍 扫描特征数据集: {feature_dataset_path}")
    print(f"   图像目录: {images_dir}")
    print(f"   信息目录: {info_dir}")
    
    # 显示过滤条件
    if bg_mode:
        print(f"   背景过滤: {bg_mode}")
    if power_filter:
        print(f"   功率过滤: {power_filter}")
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    print(f"📊 找到 {len(image_files)} 个特征图像文件")
    
    # 随机打乱图像文件列表，确保采样的多样性
    import random
    import time
    random.seed(int(time.time()))  # 使用当前时间作为随机种子
    random.shuffle(image_files)
    
    found_images = []
    valid_count = 0
    filtered_count = 0
    
    # 增加处理数量到5000，确保有足够的样本多样性
    sample_size = min(5000, len(image_files))
    print(f"🎲 随机采样 {sample_size} 个文件进行解析")
    
    for img_path in image_files[:sample_size]:
        basename = os.path.basename(img_path)
        info_path = os.path.join(info_dir, basename.replace('.jpg', '.json').replace('.png', '.json'))
        
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                
                concentration = info_data.get('concentration')
                bg_type = info_data.get('bg_type', 'unknown')
                power_value = info_data.get('power', info_data.get('power_value', 'unknown'))
                
                # 应用过滤条件
                if concentration is not None:
                    # 背景类型过滤
                    if bg_mode and bg_type != bg_mode:
                        filtered_count += 1
                        continue
                    
                    # 功率过滤
                    if power_filter and power_value != power_filter:
                        filtered_count += 1
                        continue
                    
                    # 构建完整的模式描述
                    power_suffix = f"_{power_value}" if power_value != 'unknown' else ""
                    mode_description = f"{bg_type}{power_suffix}"
                    
                    found_images.append({
                        'path': img_path,
                        'concentration': float(concentration),
                        'bg_type': bg_type,
                        'power_value': power_value,
                        'filename': basename,
                        'description': f'{mode_description}-{concentration}',
                        'dataset_type': 'feature',
                        'info_data': info_data,
                        'mode_description': mode_description
                    })
                    valid_count += 1
                    
            except Exception as e:
                continue
    
    # 按mode_description和浓度排序（但在函数外部会根据random_select参数决定是否保持随机顺序）
    found_images.sort(key=lambda x: (x['mode_description'], x['concentration']))
    
    print(f"✅ 成功解析 {len(found_images)} 个符合条件的样本 (过滤掉 {filtered_count} 个)")
    
    # 6档细分统计
    mode_stats = {}
    for img in found_images:
        mode = img['mode_description']
        if mode not in mode_stats:
            mode_stats[mode] = 0
        mode_stats[mode] += 1
    
    print(f"📊 6档细分统计:")
    for mode, count in sorted(mode_stats.items()):
        print(f"   {mode}: {count} 张")
    
    return found_images

def get_transform_for_dataset():
    """获取图像预处理变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 统一调整到224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_and_preprocess_image(image_path, transform):
    """加载并预处理图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # 记录原始尺寸
        original_size = image.size
        
        # 应用变换
        tensor = transform(image)
        
        print(f"    原始尺寸: {original_size}, 处理后: {tensor.shape}")
        
        return tensor, True, original_size
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None, False, None

def get_gradcam_target_layer(model, model_type):
    """根据模型类型获取Grad-CAM目标层 - 支持增强激光光斑CNN、ResNet50、VGG、增强CNN等"""
    if model_type.startswith('enhanced_laser_spot_cnn'):
        # 增强激光光斑CNN的最佳目标层选择策略
        print("🔍 为增强激光光斑CNN选择最佳Grad-CAM目标层...")
        
        # 优先级1: 融合卷积层（最能体现多尺度特征融合效果）
        if hasattr(model, 'fusion_conv'):
            print(f"🎯 增强激光光斑CNN目标层: fusion_conv (最佳选择)")
            return model.fusion_conv
            
        # 优先级2: 增强CBAM的空间注意力层
        elif hasattr(model, 'enhanced_cbam') and hasattr(model.enhanced_cbam, 'spatial_attention'):
            print(f"🎯 增强激光光斑CNN目标层: enhanced_cbam.spatial_attention")
            return model.enhanced_cbam.spatial_attention
            
        # 优先级3: 多尺度融合层中的最后一个
        elif hasattr(model, 'multi_scale_fusion') and len(model.multi_scale_fusion) > 0:
            target_layer = model.multi_scale_fusion[-1]  # 5x5卷积层
            print(f"🎯 增强激光光斑CNN目标层: multi_scale_fusion[-1] (5x5卷积)")
            return target_layer
            
        # 优先级4: backbone的最后一层
        elif hasattr(model, 'backbone'):
            if model_type == 'enhanced_laser_spot_cnn_vgg':
                # VGG骨干：使用features的最后一个卷积层
                for i in range(len(model.backbone) - 1, -1, -1):
                    if isinstance(model.backbone[i], nn.Conv2d):
                        print(f"🎯 增强激光光斑CNN目标层: backbone[{i}] (VGG Conv2d)")
                        return model.backbone[i]
            elif model_type == 'enhanced_laser_spot_cnn_resnet':
                # ResNet50骨干：使用layer4
                if hasattr(model.backbone, 'layer4'):
                    print(f"🎯 增强激光光斑CNN目标层: backbone.layer4")
                    return model.backbone.layer4
                else:
                    # Sequential包装的ResNet50
                    for name, layer in model.backbone.named_children():
                        if 'layer4' in name:
                            print(f"🎯 增强激光光斑CNN目标层: backbone.{name}")
                            return layer
            # 通用备选：backbone最后一层
            backbone_layers = list(model.backbone.children())
            if backbone_layers:
                print(f"🎯 增强激光光斑CNN目标层: backbone最后一层")
                return backbone_layers[-1]
        else:
            print("⚠️ 增强激光光斑CNN结构未知，使用模型倒数第二层")
            model_children = list(model.children())
            if len(model_children) >= 2:
                return model_children[-2]
            else:
                return model_children[-1] if model_children else None
            
    elif model_type == 'resnet50':
        # ResNet50使用backbone的最后一个卷积层
        if hasattr(model, 'backbone'):
            # 自定义ResNet50模型 - 选择layer4中的最后一个卷积层
            layer4 = model.backbone.layer4
            if hasattr(layer4, '__getitem__') and len(layer4) > 0:
                # layer4是Sequential，选择最后一个BasicBlock或Bottleneck
                last_block = layer4[-1]
                # 在最后一个block中找到最后一个卷积层
                if hasattr(last_block, 'conv2'):
                    print(f"🎯 ResNet50目标层: backbone.layer4[-1].conv2")
                    return last_block.conv2
                elif hasattr(last_block, 'conv3'):
                    print(f"🎯 ResNet50目标层: backbone.layer4[-1].conv3")
                    return last_block.conv3
                else:
                    print(f"🎯 ResNet50目标层: backbone.layer4[-1] (整个block)")
                    return last_block
            else:
                print(f"🎯 ResNet50目标层: backbone.layer4")
                return layer4
        else:
            # Sequential模型中的ResNet50
            layer4 = model[0].layer4
            if hasattr(layer4, '__getitem__') and len(layer4) > 0:
                last_block = layer4[-1]
                if hasattr(last_block, 'conv2'):
                    print(f"🎯 ResNet50目标层: [0].layer4[-1].conv2")
                    return last_block.conv2
                elif hasattr(last_block, 'conv3'):
                    print(f"🎯 ResNet50目标层: [0].layer4[-1].conv3")
                    return last_block.conv3
                else:
                    return last_block
            else:
                return layer4
            
    elif model_type == 'vgg':
        # VGG模型使用features的最后几层
        if hasattr(model, 'features'):
            # 找到features中的最后一个卷积层
            for i in range(len(model.features) - 1, -1, -1):
                if isinstance(model.features[i], nn.Conv2d):
                    print(f"🎯 VGG目标层: features[{i}] (Conv2d)")
                    return model.features[i]
            # 如果没找到，使用features的最后一层
            print(f"🎯 VGG目标层: features[-1] (fallback)")
            return model.features[-1]
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            # 带backbone的VGG模型
            for i in range(len(model.backbone.features) - 1, -1, -1):
                if isinstance(model.backbone.features[i], nn.Conv2d):
                    print(f"🎯 VGG目标层: backbone.features[{i}] (Conv2d)")
                    return model.backbone.features[i]
            print(f"🎯 VGG目标层: backbone.features[-1] (fallback)")
            return model.backbone.features[-1]
        else:
            print("⚠️ VGG模型结构未知，使用默认层")
            model_children = list(model.children())
            if len(model_children) >= 2:
                return model_children[-2]
            else:
                return model_children[-1] if model_children else None
            
    elif model_type == 'advanced_cnn':
        # 高级CNN使用layer4（最后的残差层）
        return model.layer4
    else:
        # 传统CNN使用conv5
        return model.conv5

def create_gradcam_visualization(model, model_name, image_info, device, model_type='traditional_cnn', use_chinese=True):
    """为单个模型和图像创建Grad-CAM可视化，支持6档细分信息"""
    image_path = image_info['path']
    concentration = image_info['concentration']
    bg_type = image_info['bg_type']
    power_value = image_info.get('power_value', 'unknown')
    description = image_info['description']
    filename = image_info['filename']
    dataset_type = image_info['dataset_type']
    mode_description = image_info.get('mode_description', f"{bg_type}_{power_value}")
    
    print(f"  Processing: {filename} ({description}) [{dataset_type}]")
    
    # 获取图像变换
    transform = get_transform_for_dataset()
    
    # 加载图像
    img_tensor, success, original_size = load_and_preprocess_image(image_path, transform)
    if not success:
        return None
    
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 获取预测 - 根据模型类型处理输出
    with torch.no_grad():
        if model_type in ['resnet50', 'vgg']:
            # ResNet50和VGG直接返回浓度值
            prediction = model(input_tensor)
        elif model_type.startswith('enhanced_laser_spot_cnn'):
            # 增强激光光斑CNN返回(浓度, 特征)元组
            prediction, _ = model(input_tensor)
        else:
            # 传统CNN和增强CNN返回(浓度, 特征)元组
            prediction, _ = model(input_tensor)
        pred_value = prediction.item()
    
    print(f"    True: {concentration}, Predicted: {pred_value:.1f}, Error: {abs(concentration - pred_value):.1f}")
    print(f"    Mode: {mode_description}, Original size: {original_size}")
    
    # 根据模型类型选择Grad-CAM目标层
    target_layer = get_gradcam_target_layer(model, model_type)
    print(f"    🎯 Grad-CAM目标层: {target_layer.__class__.__name__}")
    
    # 生成Grad-CAM（使用无hook方法避免视图修改错误）
    print(f"    🔧 使用无hook GradCAM方法（避免视图修改错误）")
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam(input_tensor, model_type)
    
    # 检查Grad-CAM是否成功
    if cam is None:
        print("    ❌ Grad-CAM生成失败，跳过此样本")
        return None
    
    # 处理可视化
    # 将tensor转换为numpy用于显示（需要detach以避免梯度错误）
    img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    # 调整CAM尺寸到224x224
    cam_resized = cv2.resize(cam, (224, 224))
    
    # 创建热力图
    heatmap = cm.jet(cam_resized)[:, :, :3]
    
    # 创建叠加图像
    overlay = 0.6 * img_np + 0.4 * heatmap
    
    return {
        'img_np': img_np,
        'heatmap': heatmap,
        'overlay': overlay,
        'cam': cam_resized,
        'concentration': concentration,
        'pred_value': pred_value,
        'error': abs(concentration - pred_value),
        'filename': filename,
        'bg_type': bg_type,
        'power_value': power_value,
        'dataset_type': dataset_type,
        'original_size': original_size,
        'mode_description': mode_description
    }

def filter_images_by_mode(images, bg_mode=None, power_filter=None):
    """根据6档细分模式过滤图像"""
    if not bg_mode and not power_filter:
        return images
    
    filtered_images = []
    for img in images:
        # 检查背景类型
        if bg_mode and img['bg_type'] != bg_mode:
            continue
        
        # 检查功率级别
        if power_filter and img['power_value'] != power_filter:
            continue
        
        filtered_images.append(img)
    
    mode_desc = ""
    if bg_mode:
        mode_desc += bg_mode
    if power_filter:
        mode_desc += f"_{power_filter}"
    
    print(f"🔍 {mode_desc}模式过滤后: {len(filtered_images)}/{len(images)} 张图像")
    return filtered_images

def parse_bg_mode(bg_mode_str):
    """解析背景模式字符串，支持6档细分"""
    if bg_mode_str == 'all':
        return None, None
    elif bg_mode_str in ['bg0', 'bg1']:
        return bg_mode_str, None
    elif '_' in bg_mode_str:
        # 6档细分模式，如 bg0_20mw
        parts = bg_mode_str.split('_')
        if len(parts) == 2:
            bg_type = parts[0]
            power_level = parts[1]
            return bg_type, power_level
    
    print(f"⚠️ 未识别的背景模式: {bg_mode_str}")
    return None, None

def save_individual_visualization(result, output_dir, sample_id, use_chinese=True):
    """保存单个样本的可视化，支持6档细分模式显示"""
    filename = result['filename'].replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    bg_type = result['bg_type']
    power_value = result.get('power_value', 'unknown')
    concentration = result['concentration']
    pred_value = result['pred_value']
    dataset_type = result['dataset_type']
    mode_description = result.get('mode_description', f"{bg_type}_{power_value}")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    title = f'样本{sample_id}: {filename} ({mode_description})[{dataset_type}]' if use_chinese else f'Sample{sample_id}: {filename} ({mode_description})[{dataset_type}]'
    fig.suptitle(title, fontsize=14)
    
    # 原始图像
    axes[0, 0].imshow(result['img_np'])
    axes[0, 0].set_title('原始图像' if use_chinese else 'Original Image')
    axes[0, 0].axis('off')
    
    # 热力图
    axes[0, 1].imshow(result['heatmap'])
    axes[0, 1].set_title('Grad-CAM热力图' if use_chinese else 'Grad-CAM Heatmap')
    axes[0, 1].axis('off')
    
    # 叠加图像
    axes[1, 0].imshow(result['overlay'])
    axes[1, 0].set_title('叠加可视化' if use_chinese else 'Overlay Visualization')
    axes[1, 0].axis('off')
    
    # 预测信息 - 更新以包含功率信息
    if use_chinese:
        info_text = f"真实浓度: {concentration:.1f}\n预测浓度: {pred_value:.1f}\n预测误差: {result['error']:.1f}\n背景类型: {bg_type}\n激光功率: {power_value}\n训练模式: {mode_description}\n数据集: {dataset_type}\n原始尺寸: {result['original_size']}"
    else:
        info_text = f"True Conc: {concentration:.1f}\nPred Conc: {pred_value:.1f}\nError: {result['error']:.1f}\nBG Type: {bg_type}\nPower: {power_value}\nMode: {mode_description}\nDataset: {dataset_type}\nOriginal: {result['original_size']}"
    
    axes[1, 1].text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')
    axes[1, 1].axis('off')
    
    # 保存
    save_name = f'sample_{sample_id:02d}_{mode_description}_{dataset_type}_{filename}_true{concentration:.0f}_pred{pred_value:.0f}.png'
    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ 保存: {save_name}")

def main():
    parser = argparse.ArgumentParser(description='增强版Grad-CAM可视化工具（支持6档细分模式）')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径（必需）')
    parser.add_argument('--dataset', type=str, choices=['feature'],
                       default='feature', help='数据集类型（当前仅支持特征数据集）')
    parser.add_argument('--bg_mode', type=str, 
                       choices=['bg0', 'bg1', 'all', 'bg0_20mw', 'bg0_100mw', 'bg0_400mw', 
                               'bg1_20mw', 'bg1_100mw', 'bg1_400mw'],
                       default='bg0', help='背景模式（支持6档细分）')
    parser.add_argument('--max_samples', type=int, default=20,
                       help='最大样本数')
    parser.add_argument('--feature_dataset', type=str, default=None,
                       help='指定特征数据集路径')
    parser.add_argument('--random_select', action='store_true',
                       help='随机选择样本（而非按浓度顺序选择）')
    
    args = parser.parse_args()
    
    print("=== 增强版Grad-CAM可视化分析（6档细分支持）===")
    print(f"模型路径: {args.model_path}")
    print(f"数据集类型: {args.dataset}")
    print(f"背景模式: {args.bg_mode}")
    print(f"最大样本数: {args.max_samples}")
    print(f"随机选择: {'是' if args.random_select else '否（按浓度顺序）'}")
    
    # 解析6档细分模式
    bg_type, power_filter = parse_bg_mode(args.bg_mode)
    if bg_type:
        print(f"   背景类型: {bg_type}")
    if power_filter:
        print(f"   功率级别: {power_filter}")
    if not bg_type and not power_filter:
        print(f"   处理所有模式的数据")
    
    # 验证模型路径
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        return
    
    # 设置中文字体
    chinese_font = setup_chinese_font()
    use_chinese = chinese_font is not None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    try:
        # 加载模型
        model, model_type = load_model_safe(args.model_path, device)
        
        # 生成唯一的模型名称（包含父目录信息）
        model_path_obj = Path(args.model_path)
        model_filename = model_path_obj.stem  # 不带扩展名的文件名
        parent_dir = model_path_obj.parent.name
        
        # 如果文件名是通用的（如best_model），则使用父目录信息
        if model_filename.lower() in ['best_model', 'model', 'final_model']:
            # 从父目录提取有用信息
            if parent_dir and parent_dir != '.':
                # 提取时间戳和其他标识信息
                import re
                # 尝试提取时间戳
                timestamp_match = re.search(r'(\d{8}_\d{6})', parent_dir)
                if timestamp_match:
                    model_name = f"{parent_dir}_{model_filename}"
                else:
                    model_name = f"{parent_dir}_{model_filename}"
            else:
                model_name = model_filename
        else:
            model_name = model_filename
            
        print(f"📋 使用模型类型: {model_type}")
        print(f"📋 模型标识: {model_name}")
        
        # 扫描特征数据集
        print("\n=== 扫描特征数据集 ===")
        feature_images = find_feature_dataset_images(args.feature_dataset, bg_type, power_filter)
        
        if not feature_images:
            print("❌ 未找到符合条件的图像")
            return
        
        # 选择样本 - 新增随机选择逻辑
        if args.random_select:
            import random
            random.seed(42)  # 设置随机种子确保可重现
            # 先随机打乱，再选择前max_samples个
            random.shuffle(feature_images)
            selected_images = feature_images[:args.max_samples]
            print(f"\n🎲 随机选择处理 {len(selected_images)} 张图像")
        else:
            selected_images = feature_images[:args.max_samples]
            print(f"\n📊 按顺序选择处理 {len(selected_images)} 张图像")
        
        # 创建输出目录
        suffix = 'random' if args.random_select else 'ordered'
        mode_suffix = args.bg_mode if args.bg_mode != 'all' else 'all'
        output_dir = f'enhanced_gradcam_analysis_{model_name}_{mode_suffix}_{args.dataset}_{suffix}'
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")
        
        # 处理图像
        successful_visualizations = []
        
        for i, image_info in enumerate(selected_images, 1):
            print(f"\n📸 样本 {i}/{len(selected_images)}")
            
            result = create_gradcam_visualization(
                model, model_name, image_info, device, model_type, use_chinese
            )
            
            if result is not None:
                successful_visualizations.append(result)
                save_individual_visualization(result, output_dir, i, use_chinese)
            else:
                print(f"    ❌ 可视化生成失败")
        
        print(f"\n🎉 分析完成！共处理 {len(successful_visualizations)} 个样本")
        print(f"📁 查看结果: {output_dir}/")
        
        # 统计信息
        print(f"   模式过滤: {args.bg_mode}")
        print(f"   特征数据集样本: {len(successful_visualizations)}")
        avg_error = np.mean([r['error'] for r in successful_visualizations])
        print(f"   平均预测误差: {avg_error:.2f}")
        
        # 显示选择的浓度分布
        concentrations = [r['concentration'] for r in successful_visualizations]
        print(f"   浓度分布: {min(concentrations):.1f} - {max(concentrations):.1f}")
        print(f"   选择策略: {'随机采样' if args.random_select else '按浓度顺序'}")
        
        # 6档细分统计
        if successful_visualizations:
            mode_stats = {}
            for result in successful_visualizations:
                mode = result.get('mode_description', f"{result['bg_type']}_{result.get('power_value', 'unknown')}")
                if mode not in mode_stats:
                    mode_stats[mode] = 0
                mode_stats[mode] += 1
            
            print(f"   6档细分分布:")
            for mode, count in sorted(mode_stats.items()):
                print(f"     {mode}: {count} 张")
            
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 