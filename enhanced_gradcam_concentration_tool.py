#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强Grad-CAM浓度可视化工具
支持按悬浮物浓度从数据集随机选取对应浓度图像生成Grad-CAM可视化图像
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# 添加路径
sys.path.append('src')
sys.path.append('cloud_vgg_training_package')

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GradCAMConcentrationVisualizer:
    """按浓度的Grad-CAM可视化器"""
    
    def __init__(self, output_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"gradcam_concentration_analysis_{timestamp}"
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🔍 Grad-CAM浓度可视化工具初始化")
        print(f"   设备: {self.device}")
        print(f"   输出目录: {self.output_dir}")
        
        # Grad-CAM相关
        self.gradients = None
        self.activations = None
    
    def register_hooks(self, model, target_layer_name):
        """注册钩子函数"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # 找到目标层
        target_layer = None
        for name, module in model.named_modules():
            if target_layer_name in name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"未找到目标层: {target_layer_name}")
        
        # 注册钩子
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        
        print(f"   已注册钩子到层: {target_layer_name}")
        return target_layer
    
    def generate_gradcam(self, model, image, target_layer_name, class_idx=None):
        """生成Grad-CAM热力图"""
        # 注册钩子
        self.register_hooks(model, target_layer_name)
        
        # 前向传播
        model.eval()
        image_tensor = image.unsqueeze(0).to(self.device)
        image_tensor.requires_grad_(True)
        
        output = model(image_tensor)
        
        # 处理可能的tuple输出
        if isinstance(output, tuple):
            output = output[0]
        
        # 如果是回归任务，直接对输出求梯度
        if class_idx is None:
            score = output[0]  # 回归输出的第一个值
        else:
            score = output[0][class_idx]
        
        # 反向传播
        model.zero_grad()
        score.backward(retain_graph=True)
        
        # 获取梯度和激活
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # 计算权重
        weights = np.mean(gradients, axis=(1, 2))
        
        # 生成Grad-CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU激活
        cam = np.maximum(cam, 0)
        
        # 归一化
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def apply_colormap_on_image(self, org_im, activation, colormap_name='jet'):
        """将热力图应用到原始图像上"""
        # 将激活图调整到原始图像大小
        heatmap = cv2.resize(activation, (org_im.shape[1], org_im.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加到原始图像
        superimposed_img = heatmap * 0.4 + org_im * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img, heatmap
    
    def load_dataset_by_concentration(self, dataset_path, bg_mode='all'):
        """按浓度加载数据集"""
        print(f"📂 加载数据集: {dataset_path}")
        
        # 加载数据集
        from feature_dataset_loader import create_feature_dataloader
        
        bg_filter, power_filter = self._parse_bg_mode(bg_mode)
        
        # 创建数据加载器（不打乱顺序）
        dataloader, dataset = create_feature_dataloader(
            feature_dataset_path=dataset_path,
            batch_size=1,
            shuffle=False,
            bg_type=bg_filter,
            power_filter=power_filter
        )
        
        # 按浓度分组
        concentration_groups = defaultdict(list)
        
        print(f"   正在按浓度分组... (共{len(dataset)}个样本)")
        
        for idx, (image, concentration, metadata) in enumerate(dataset):
            # 将浓度四舍五入到最近的整数
            conc_rounded = round(float(concentration))
            
            # 获取原始图像路径
            original_image_path = dataset.image_files[idx] if hasattr(dataset, 'image_files') else None
            
            concentration_groups[conc_rounded].append({
                'index': idx,
                'image': image,
                'concentration': float(concentration),
                'metadata': metadata,
                'original_image_path': original_image_path  # 添加原始图像路径
            })
            
            if (idx + 1) % 1000 == 0:
                print(f"   进度: {idx + 1}/{len(dataset)}")
        
        print(f"   完成分组，共{len(concentration_groups)}个浓度级别")
        
        # 统计每个浓度的样本数
        conc_stats = {}
        for conc, samples in concentration_groups.items():
            conc_stats[conc] = len(samples)
        
        # 显示浓度分布
        sorted_concs = sorted(conc_stats.keys())
        print(f"   浓度范围: {min(sorted_concs)} - {max(sorted_concs)} mg/L")
        print(f"   浓度级别数: {len(sorted_concs)}")
        
        return concentration_groups, conc_stats
    
    def load_original_image(self, image_path):
        """加载原始图像（未经预处理）"""
        try:
            from PIL import Image
            original_image = Image.open(image_path).convert('RGB')
            # 转换为numpy数组
            original_np = np.array(original_image)
            return original_np
        except Exception as e:
            print(f"   警告: 无法加载原始图像 {image_path}: {e}")
            return None
    
    def select_samples_by_concentration(self, concentration_groups, 
                                      min_conc, max_conc, interval, 
                                      samples_per_conc=10):
        """按浓度区间选择样本"""
        print(f"\n🎯 按浓度选择样本:")
        print(f"   浓度范围: {min_conc} - {max_conc} mg/L")
        print(f"   间隔: {interval} mg/L")
        print(f"   每个浓度选择: {samples_per_conc} 个样本")
        
        selected_samples = []
        target_concentrations = list(range(min_conc, max_conc + 1, interval))
        
        for target_conc in target_concentrations:
            # 找到最接近的浓度级别
            available_concs = list(concentration_groups.keys())
            closest_conc = min(available_concs, 
                             key=lambda x: abs(x - target_conc))
            
            # 检查是否在合理范围内
            if abs(closest_conc - target_conc) <= interval / 2:
                samples = concentration_groups[closest_conc]
                
                # 随机选择样本
                num_to_select = min(samples_per_conc, len(samples))
                selected = random.sample(samples, num_to_select)
                
                for sample in selected:
                    sample['target_concentration'] = target_conc
                    sample['actual_concentration'] = closest_conc
                
                selected_samples.extend(selected)
                
                print(f"   {target_conc} mg/L: 选择了{num_to_select}个样本 (实际浓度:{closest_conc} mg/L)")
            else:
                print(f"   {target_conc} mg/L: 未找到合适的样本 (最近:{closest_conc} mg/L)")
        
        print(f"\n   总共选择了{len(selected_samples)}个样本")
        return selected_samples
    
    def visualize_concentration_samples(self, model, model_name, target_layer_name,
                                      selected_samples, save_individual=True):
        """可视化选定的浓度样本"""
        print(f"\n🎨 生成Grad-CAM可视化...")
        
        # 按目标浓度分组
        conc_groups = defaultdict(list)
        for sample in selected_samples:
            conc_groups[sample['target_concentration']].append(sample)
        
        all_visualizations = []
        
        for target_conc in sorted(conc_groups.keys()):
            samples = conc_groups[target_conc]
            print(f"   处理浓度 {target_conc} mg/L: {len(samples)} 个样本")
            
            conc_visualizations = []
            
            for i, sample in enumerate(samples):
                try:
                    # 生成Grad-CAM
                    cam = self.generate_gradcam(
                        model, sample['image'], target_layer_name
                    )
                    
                    # 转换预处理后的图像格式（用于生成热力图）
                    processed_image_np = sample['image'].permute(1, 2, 0).numpy()
                    processed_image_np = (processed_image_np * 255).astype(np.uint8)
                    
                    # 加载原始图像
                    original_image_np = None
                    if sample.get('original_image_path'):
                        original_image_np = self.load_original_image(sample['original_image_path'])
                    
                    # 如果无法加载原始图像，使用预处理后的图像作为替代
                    if original_image_np is None:
                        original_image_np = processed_image_np
                        print(f"     样本 {i}: 使用预处理图像作为原始图像")
                    
                    # 应用热力图到预处理图像（用于Grad-CAM生成）
                    superimposed_processed, heatmap = self.apply_colormap_on_image(
                        processed_image_np, cam
                    )
                    
                    # 应用热力图到原始图像（用于展示）
                    superimposed_original, _ = self.apply_colormap_on_image(
                        original_image_np, cam
                    )
                    
                    # 预测浓度
                    with torch.no_grad():
                        model.eval()
                        image_tensor = sample['image'].unsqueeze(0).to(self.device)
                        output = model(image_tensor)
                        if isinstance(output, tuple):
                            output = output[0]
                        predicted_conc = output.item()
                    
                    visualization = {
                        'target_concentration': target_conc,
                        'actual_concentration': sample['actual_concentration'],
                        'predicted_concentration': predicted_conc,
                        'original_image': original_image_np,  # 真正的原始图像
                        'processed_image': processed_image_np,  # 预处理后的图像
                        'heatmap': heatmap,
                        'superimposed_processed': superimposed_processed,  # 热力图+预处理图像
                        'superimposed_original': superimposed_original,    # 热力图+原始图像
                        'sample_index': sample['index'],
                        'image_path': sample.get('original_image_path', 'Unknown')
                    }
                    
                    conc_visualizations.append(visualization)
                    
                    # 保存单个可视化
                    if save_individual:
                        self._save_individual_visualization(
                            visualization, model_name, target_conc, i
                        )
                    
                except Exception as e:
                    print(f"     样本 {i} 处理失败: {e}")
                    continue
            
            if conc_visualizations:
                # 创建浓度网格
                grid_path = self._create_concentration_grid(
                    conc_visualizations, model_name, target_conc
                )
                
                all_visualizations.extend(conc_visualizations)
        
        # 创建总览可视化
        if all_visualizations:
            overview_path = self._create_overview_visualization(
                all_visualizations, model_name
            )
            print(f"   ✅ 总览可视化已保存: {overview_path}")
        
        return all_visualizations
    
    def _save_individual_visualization(self, viz, model_name, target_conc, sample_idx):
        """保存单个可视化图像"""
        # 创建子目录
        conc_dir = os.path.join(self.output_dir, f"concentration_{target_conc}mg_L")
        os.makedirs(conc_dir, exist_ok=True)
        
        # 创建组合图（2行2列）
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始数据图像
        axes[0, 0].imshow(viz['original_image'])
        axes[0, 0].set_title('原始数据图像', fontsize=12)
        axes[0, 0].axis('off')
        
        # 预处理后图像
        axes[0, 1].imshow(viz['processed_image'])
        axes[0, 1].set_title('预处理后图像', fontsize=12)
        axes[0, 1].axis('off')
        
        # Grad-CAM热力图
        axes[1, 0].imshow(viz['heatmap'])
        axes[1, 0].set_title('Grad-CAM热力图', fontsize=12)
        axes[1, 0].axis('off')
        
        # 原始图像+热力图叠加
        axes[1, 1].imshow(viz['superimposed_original'])
        axes[1, 1].set_title('原始图像+热力图', fontsize=12)
        axes[1, 1].axis('off')
        
        # 添加详细信息
        info_text = (f"目标浓度: {viz['target_concentration']} mg/L\n"
                    f"实际浓度: {viz['actual_concentration']:.1f} mg/L\n"
                    f"预测浓度: {viz['predicted_concentration']:.1f} mg/L\n"
                    f"图像路径: {os.path.basename(viz['image_path'])}")
        
        fig.suptitle(f"{model_name} - 样本 {sample_idx + 1}\n{info_text}", 
                    fontsize=14, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # 为标题留出空间
        
        # 保存
        filename = f"sample_{sample_idx + 1:02d}_{target_conc}mg_L_detailed.png"
        save_path = os.path.join(conc_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 同时保存简化版本（只有原始图像、热力图、叠加图像）
        fig_simple, axes_simple = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始数据图像
        axes_simple[0].imshow(viz['original_image'])
        axes_simple[0].set_title('原始数据图像')
        axes_simple[0].axis('off')
        
        # 热力图
        axes_simple[1].imshow(viz['heatmap'])
        axes_simple[1].set_title('Grad-CAM热力图')
        axes_simple[1].axis('off')
        
        # 叠加图像
        axes_simple[2].imshow(viz['superimposed_original'])
        axes_simple[2].set_title('原始图像+热力图')
        axes_simple[2].axis('off')
        
        # 简化信息
        info_simple = (f"目标: {viz['target_concentration']} mg/L | "
                      f"实际: {viz['actual_concentration']:.1f} mg/L | "
                      f"预测: {viz['predicted_concentration']:.1f} mg/L")
        
        fig_simple.suptitle(f"{model_name} - 样本 {sample_idx + 1}\n{info_simple}", 
                           fontsize=12)
        
        plt.tight_layout()
        
        # 保存简化版本
        filename_simple = f"sample_{sample_idx + 1:02d}_{target_conc}mg_L_simple.png"
        save_path_simple = os.path.join(conc_dir, filename_simple)
        plt.savefig(save_path_simple, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_concentration_grid(self, visualizations, model_name, target_conc):
        """创建单个浓度的网格图"""
        if not visualizations:
            return None # Changed to return None
        
        num_samples = len(visualizations)
        cols = min(5, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 3, rows * 8))
        
        if rows == 1 and cols == 1:
            axes = axes.reshape(3, 1)
        elif rows == 1:
            axes = axes.reshape(3, cols)
        elif cols == 1:
            axes = axes.reshape(rows * 3, 1)
        
        for i, viz in enumerate(visualizations):
            col = i % cols
            row_base = (i // cols) * 3
            
            # 原始图像
            if rows == 1:
                axes[0, col].imshow(viz['original_image'])
                axes[0, col].set_title(f'原始 #{i+1}')
                axes[0, col].axis('off')
                
                # 热力图
                axes[1, col].imshow(viz['heatmap'])
                axes[1, col].set_title(f'热力图 #{i+1}')
                axes[1, col].axis('off')
                
                # 叠加图像
                axes[2, col].imshow(viz['superimposed_original']) # Changed to superimposed_original
                axes[2, col].set_title(f'叠加 #{i+1}\n预测:{viz["predicted_concentration"]:.1f}')
                axes[2, col].axis('off')
            else:
                axes[row_base, col].imshow(viz['original_image'])
                axes[row_base, col].set_title(f'原始 #{i+1}')
                axes[row_base, col].axis('off')
                
                # 热力图
                axes[row_base + 1, col].imshow(viz['heatmap'])
                axes[row_base + 1, col].set_title(f'热力图 #{i+1}')
                axes[row_base + 1, col].axis('off')
                
                # 叠加图像
                axes[row_base + 2, col].imshow(viz['superimposed_original']) # Changed to superimposed_original
                axes[row_base + 2, col].set_title(f'叠加 #{i+1}\n预测:{viz["predicted_concentration"]:.1f}')
                axes[row_base + 2, col].axis('off')
        
        # 隐藏多余的子图
        total_subplots = rows * 3 * cols
        used_subplots = len(visualizations) * 3
        for i in range(used_subplots, total_subplots):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.suptitle(f"{model_name} - {target_conc} mg/L 浓度样本", fontsize=16)
        plt.tight_layout()
        
        # 保存
        filename = f"concentration_{target_conc}mg_L_grid.png"
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     网格图已保存: {filename}")
        return save_path # Added return statement
    
    def _create_overview_visualization(self, all_visualizations, model_name):
        """创建总览可视化"""
        if not all_visualizations:
            return None # Changed to return None
        
        # 按浓度分组
        conc_groups = defaultdict(list)
        for viz in all_visualizations:
            conc_groups[viz['target_concentration']].append(viz)
        
        # 每个浓度选择一个代表性样本
        representative_samples = []
        for target_conc in sorted(conc_groups.keys()):
            samples = conc_groups[target_conc]
            # 选择预测最准确的样本
            best_sample = min(samples, 
                            key=lambda x: abs(x['predicted_concentration'] - x['actual_concentration']))
            representative_samples.append(best_sample)
        
        # 创建总览图
        num_concs = len(representative_samples)
        cols = min(6, num_concs)
        rows = (num_concs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))
        
        if rows == 1 and cols == 1:
            axes = axes.reshape(2, 1)
        elif rows == 1:
            axes = axes.reshape(2, cols)
        elif cols == 1:
            axes = axes.reshape(rows * 2, 1)
        
        for i, viz in enumerate(representative_samples):
            col = i % cols
            row_base = (i // cols) * 2
            
            # 原始图像
            if rows == 1:
                axes[0, col].imshow(viz['original_image'])
                axes[0, col].set_title(f'{viz["target_concentration"]} mg/L\n原始图像')
                axes[0, col].axis('off')
                
                # 叠加图像
                axes[1, col].imshow(viz['superimposed_original']) # Changed to superimposed_original
                axes[1, col].set_title(f'Grad-CAM\n预测:{viz["predicted_concentration"]:.1f}')
                axes[1, col].axis('off')
            else:
                axes[row_base, col].imshow(viz['original_image'])
                axes[row_base, col].set_title(f'{viz["target_concentration"]} mg/L\n原始图像')
                axes[row_base, col].axis('off')
                
                # 叠加图像
                axes[row_base + 1, col].imshow(viz['superimposed_original']) # Changed to superimposed_original
                axes[row_base + 1, col].set_title(f'Grad-CAM\n预测:{viz["predicted_concentration"]:.1f}')
                axes[row_base + 1, col].axis('off')
        
        # 隐藏多余的子图
        total_subplots = rows * 2 * cols
        used_subplots = len(representative_samples) * 2
        for i in range(used_subplots, total_subplots):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.suptitle(f"{model_name} - 不同浓度的Grad-CAM可视化总览", fontsize=16)
        plt.tight_layout()
        
        # 保存
        overview_path = os.path.join(self.output_dir, "concentration_overview.png")
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   总览图已保存: concentration_overview.png")
        return overview_path # Added return statement
    
    def _parse_bg_mode(self, bg_mode):
        """解析背景模式"""
        if '_' in bg_mode:
            parts = bg_mode.split('_')
            bg_filter = parts[0]
            power_filter = parts[1]
        else:
            bg_filter = bg_mode if bg_mode != 'all' else None
            power_filter = None
        
        return bg_filter, power_filter
    
    def _load_checkpoint_safely(self, model_path):
        """安全加载checkpoint"""
        try:
            # 首先尝试使用weights_only=True（安全模式）
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            # 如果安全模式失败，使用weights_only=False（兼容旧格式）
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        return checkpoint
    
    def load_model(self, model_type, model_path):
        """加载模型"""
        print(f"🔧 加载{model_type}模型: {model_path}")
        
        checkpoint_to_use = None  # 用于存储预加载的checkpoint
        
        if model_type == 'baseline_cnn':
            from cnn_model import CNNFeatureExtractor
            model = CNNFeatureExtractor().to(self.device)
            target_layer = 'last_conv'  # 基线CNN模型的最后一个卷积层
        elif model_type == 'enhanced_cnn':
            from enhanced_laser_spot_cnn import create_enhanced_laser_spot_model
            model = create_enhanced_laser_spot_model(backbone='resnet18').to(self.device)
            target_layer = 'backbone.layer4'
        elif model_type in ['cloud_resnet50', 'adaptive_resnet50']:
            # 首先尝试加载权重来检测模型类型
            try:
                # 先加载checkpoint检查权重键
                checkpoint = self._load_checkpoint_safely(model_path)
                
                # 检查是否包含注意力模块权重
                has_attention = any('attention_modules' in key for key in checkpoint.get('model_state_dict', checkpoint).keys())
                
                if has_attention or model_type == 'adaptive_resnet50':
                    # 使用自适应ResNet50
                    try:
                        from compatible_adaptive_resnet50 import CompatibleAdaptiveResNet50
                        model = CompatibleAdaptiveResNet50().to(self.device)
                        target_layer = 'backbone.layer4'
                        print(f"   使用自适应ResNet50模型")
                    except ImportError:
                        print(f"   警告: 无法导入CompatibleAdaptiveResNet50，回退到标准ResNet50")
                        # 回退到标准ResNet50
                        import torchvision.models as models
                        model = models.resnet50(pretrained=False)
                        model.fc = torch.nn.Linear(model.fc.in_features, 1)
                        model = model.to(self.device)
                        target_layer = 'layer4'
                else:
                    # 使用标准ResNet50
                    import torchvision.models as models
                    model = models.resnet50(pretrained=False)
                    model.fc = torch.nn.Linear(model.fc.in_features, 1)
                    model = model.to(self.device)
                    target_layer = 'layer4'
                    print(f"   使用标准ResNet50模型")
                
                # 直接使用已加载的checkpoint
                checkpoint_to_use = checkpoint
                
            except Exception as e:
                print(f"   模型检测失败: {e}")
                # 默认使用标准ResNet50
                import torchvision.models as models
                model = models.resnet50(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 1)
                model = model.to(self.device)
                target_layer = 'layer4'
                checkpoint_to_use = None  # 稍后重新加载
 
        elif model_type == 'cloud_vgg':
            from vgg_regression import VGGRegressionCBAM
            model = VGGRegressionCBAM().to(self.device)
            target_layer = 'features'
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载权重
        if checkpoint_to_use is None:
            # 如果没有预加载checkpoint，现在加载
            checkpoint_to_use = self._load_checkpoint_safely(model_path)
        
        try:
            if isinstance(checkpoint_to_use, dict) and 'model_state_dict' in checkpoint_to_use:
                model.load_state_dict(checkpoint_to_use['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint_to_use, strict=False)
            print(f"   模型权重加载成功（使用strict=False模式）")
        except Exception as e:
            print(f"   权重加载失败: {e}")
            raise
        
        model.eval()
        
        print(f"   模型加载成功，目标层: {target_layer}")
        return model, target_layer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强Grad-CAM浓度可视化工具')
    
    # 基本参数
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['baseline_cnn', 'enhanced_cnn', 'cloud_resnet50', 'cloud_vgg', 'adaptive_resnet50'],
                       help='模型类型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='特征数据集路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    
    # 浓度选择参数
    parser.add_argument('--min_concentration', type=int, default=0,
                       help='最小浓度 (mg/L)')
    parser.add_argument('--max_concentration', type=int, default=1000,
                       help='最大浓度 (mg/L)')
    parser.add_argument('--concentration_interval', type=int, default=100,
                       help='浓度间隔 (mg/L)')
    parser.add_argument('--samples_per_concentration', type=int, default=10,
                       help='每个浓度选择的样本数')
    
    # 其他参数
    parser.add_argument('--bg_mode', type=str, default='all',
                       help='背景模式 (bg0, bg1, all, bg0_20mw, etc.)')
    parser.add_argument('--target_layer', type=str, default=None,
                       help='目标层名称 (自动检测)')
    parser.add_argument('--save_individual', action='store_true',
                       help='保存单个样本的可视化')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    print("🔍 增强Grad-CAM浓度可视化工具")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建可视化器
    visualizer = GradCAMConcentrationVisualizer(args.output_dir)
    
    # 自动检测数据集
    if args.dataset_path is None:
        sys.path.append('src')
        from feature_dataset_loader import detect_feature_datasets
        datasets = detect_feature_datasets()
        if datasets:
            args.dataset_path = datasets[-1]['path']
            print(f"🔍 自动检测到数据集: {args.dataset_path}")
        else:
            raise FileNotFoundError("未找到特征数据集，请使用--dataset_path指定")
    
    # 加载模型
    model, target_layer = visualizer.load_model(args.model_type, args.model_path)
    
    if args.target_layer:
        target_layer = args.target_layer
    
    # 按浓度加载数据集
    concentration_groups, conc_stats = visualizer.load_dataset_by_concentration(
        args.dataset_path, args.bg_mode
    )
    
    # 选择样本
    selected_samples = visualizer.select_samples_by_concentration(
        concentration_groups,
        args.min_concentration,
        args.max_concentration, 
        args.concentration_interval,
        args.samples_per_concentration
    )
    
    if not selected_samples:
        print("❌ 未选择到任何样本")
        return
    
    # 生成可视化
    model_name = f"{args.model_type}_{Path(args.model_path).stem}"
    visualizations = visualizer.visualize_concentration_samples(
        model, model_name, target_layer, selected_samples, args.save_individual
    )
    
    # 保存统计信息
    stats = {
        'model_type': args.model_type,
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
        'concentration_range': f"{args.min_concentration}-{args.max_concentration}",
        'interval': args.concentration_interval,
        'samples_per_concentration': args.samples_per_concentration,
        'total_selected_samples': len(selected_samples),
        'total_visualizations': len(visualizations),
        'concentration_stats': conc_stats,
        'bg_mode': args.bg_mode
    }
    
    stats_path = os.path.join(visualizer.output_dir, "visualization_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Grad-CAM浓度可视化完成!")
    print(f"   生成了{len(visualizations)}个可视化")
    print(f"   结果保存在: {visualizer.output_dir}")
    print(f"   统计信息: {stats_path}")

if __name__ == "__main__":
    main() 