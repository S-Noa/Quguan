#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练过程监控模块
包含Grad-CAM可视化和水印区域关注度检查
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from PIL import Image
import time
from font_utils import auto_setup_font, get_labels, CHINESE_SUPPORTED

class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self, model_type='cnn', save_dir='monitoring_results', device='cuda'):
        """
        初始化监控器
        
        Args:
            model_type: 模型类型 ('cnn' 或 'vgg')
            save_dir: 监控结果保存目录
            device: 计算设备
        """
        self.model_type = model_type
        self.save_dir = save_dir
        self.device = device
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置字体
        self.font_available = CHINESE_SUPPORTED
        self.labels = get_labels(use_chinese=self.font_available)
        
        # 水印区域定义（左下角）
        self.watermark_regions = {
            'small': (0.75, 0.85, 0.95, 0.95),   # 水印区域-小
            'medium': (0.70, 0.80, 1.0, 1.0),    # 水印区域-中  
            'large': (0.65, 0.75, 1.0, 1.0)      # 水印区域-大
        }
        
        # 中心区域定义（光斑主要区域）
        self.center_region = (0.35, 0.35, 0.65, 0.65)  # 中心区域 (x1, y1, x2, y2)
        
        # 监控历史
        self.attention_history = []
        self.epoch_gradcams = []
        
        print(f"✅ 训练监控器初始化完成")
        print(f"   模型类型: {model_type}")
        print(f"   保存目录: {save_dir}")
        print(f"   字体支持: {'中文' if self.font_available else '英文'}")
        
    def get_target_layer(self, model):
        """获取目标层用于Grad-CAM"""
        if self.model_type == 'cnn':
            # CNN模型：使用最后一个卷积层
            target_layer = None
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
            return target_layer
        elif self.model_type == 'vgg':
            # VGG模型：使用CBAM层
            target_layer = None
            for name, module in model.features.named_modules():
                if 'cbam_5' in name or 'cbam' in name:
                    target_layer = module
                    break
            if target_layer is None:
                # 如果没有CBAM层，使用最后一个卷积层
                for name, module in model.features.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
            return target_layer
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def generate_gradcam(self, model, input_tensor, target_layer):
        """生成Grad-CAM热力图"""
        # 保存当前模型的训练状态
        original_training = model.training
        
        # 创建Grad-CAM对象
        grad_cam = GradCAM(model, target_layer)
        
        # 生成热力图
        try:
            cam = grad_cam(input_tensor)
            return cam
        except Exception as e:
            print(f"⚠️ Grad-CAM生成失败: {e}")
            return None
        finally:
            # 清理hooks
            grad_cam.remove_hooks()
            # 恢复模型原始状态
            model.train(original_training)
    
    def analyze_attention_regions(self, cam, img_size=(224, 224)):
        """分析注意力区域分布"""
        if cam is None:
            return None
        
        h, w = img_size
        cam_resized = cv2.resize(cam, (w, h))
        
        # 计算不同区域的平均注意力
        regions_attention = {}
        
        # 水印区域
        for region_name, (x1_ratio, y1_ratio, x2_ratio, y2_ratio) in self.watermark_regions.items():
            x1, y1 = int(x1_ratio * w), int(y1_ratio * h)
            x2, y2 = int(x2_ratio * w), int(y2_ratio * h)
            region_attention = cam_resized[y1:y2, x1:x2].mean()
            regions_attention[f'watermark_{region_name}'] = region_attention
        
        # 中心区域
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = self.center_region
        x1, y1 = int(x1_ratio * w), int(y1_ratio * h)
        x2, y2 = int(x2_ratio * w), int(y2_ratio * h)
        center_attention = cam_resized[y1:y2, x1:x2].mean()
        regions_attention['center'] = center_attention
        
        # 计算注意力比例
        attention_ratios = {}
        for region_name in ['small', 'medium', 'large']:
            watermark_key = f'watermark_{region_name}'
            if center_attention > 0:
                ratio = regions_attention[watermark_key] / center_attention
                attention_ratios[f'{region_name}_ratio'] = ratio
            else:
                attention_ratios[f'{region_name}_ratio'] = 0.0
        
        return {
            'regions_attention': regions_attention,
            'attention_ratios': attention_ratios,
            'cam_resized': cam_resized
        }
    
    def check_watermark_attention(self, analysis_result, threshold=0.3):
        """检查水印区域关注度是否过高"""
        if analysis_result is None:
            return False, "分析结果为空"
        
        attention_ratios = analysis_result['attention_ratios']
        warnings = []
        
        for region_name in ['small', 'medium', 'large']:
            ratio_key = f'{region_name}_ratio'
            ratio = attention_ratios.get(ratio_key, 0.0)
            
            if ratio > threshold:
                warnings.append(f"水印区域({region_name})关注度过高: {ratio:.3f} > {threshold}")
        
        has_warning = len(warnings) > 0
        warning_msg = "; ".join(warnings) if warnings else "水印关注度正常"
        
        return has_warning, warning_msg
    
    def visualize_gradcam_with_regions(self, model, sample_image, epoch, save_name=None):
        """生成带区域标记的Grad-CAM可视化"""
        if save_name is None:
            save_name = f'{self.model_type}_gradcam_epoch_{epoch}'
        
        # 🔧 修复图像处理逻辑
        if isinstance(sample_image, torch.Tensor):
            # 如果输入是tensor
            if sample_image.dim() == 4:  # 批次维度
                input_tensor = sample_image.to(self.device)
                img_for_display = sample_image[0]  # 取第一个
            elif sample_image.dim() == 3:  # 单张图像
                input_tensor = sample_image.unsqueeze(0).to(self.device)
                img_for_display = sample_image
            else:
                raise ValueError(f"不支持的tensor维度: {sample_image.dim()}")
            
            # 准备显示用的图像
            if img_for_display.dim() == 3:
                # 从 (C, H, W) 转换为 (H, W, C)
                img_np = img_for_display.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = img_for_display.cpu().numpy()
        else:
            # 如果输入是PIL图像或numpy数组
            if isinstance(sample_image, Image.Image):
                img_np = np.array(sample_image)
            else:
                img_np = sample_image
            
            # 转换为tensor用于模型输入
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            else:
                raise ValueError("不支持的图像格式")
            
            input_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 生成Grad-CAM
        target_layer = self.get_target_layer(model)
        if target_layer is None:
            print(f"⚠️ 警告：未找到合适的目标层，跳过Grad-CAM生成")
            return None
        
        cam = self.generate_gradcam(model, input_tensor, target_layer)
        if cam is None:
            print(f"⚠️ 跳过epoch {epoch}的Grad-CAM可视化")
            return None
        
        # 分析注意力区域
        analysis_result = self.analyze_attention_regions(cam)
        if analysis_result is None:
            print(f"⚠️ 跳过epoch {epoch}的注意力分析")
            return None
        
        # 检查水印关注度
        has_warning, warning_msg = self.check_watermark_attention(analysis_result)
        
        # 🔧 修复图像显示逻辑
        # 确保图像是正确的格式 (H, W, C) 且值在 [0, 255]
        if img_np.max() <= 1.0:
            # 如果图像已归一化，需要反归一化
            if img_np.shape[2] == 3:  # RGB图像
                # ImageNet标准反归一化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        # 调整热力图大小
        h, w = img_np.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # 🎨 创建可视化 - 使用修复后的字体
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        title = f'Epoch {epoch} Grad-CAM 监控分析' if self.font_available else f'Epoch {epoch} Grad-CAM Analysis'
        fig.suptitle(title, fontsize=16)
        
        # 原始图像
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(self.labels['original_image'])
        axes[0, 0].axis('off')
        
        # 热力图
        heatmap = cm.jet(cam_resized)[:, :, :3]
        axes[0, 1].imshow(heatmap)
        axes[0, 1].set_title(self.labels['gradcam_heatmap'])
        axes[0, 1].axis('off')
        
        # 叠加图像
        overlay = cv2.addWeighted(img_np, 0.6, (heatmap * 255).astype(np.uint8), 0.4, 0)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title(self.labels['overlay_visualization'])
        axes[1, 0].axis('off')
        
        # 添加区域标记
        self._add_region_markers(axes[1, 0], w, h)
        
        # 注意力分析结果
        analysis_text = self._format_analysis_text(analysis_result, has_warning, warning_msg)
        axes[1, 1].text(0.05, 0.95, analysis_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f'{save_name}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 记录历史
        self.attention_history.append({
            'epoch': epoch,
            'analysis': analysis_result,
            'has_warning': has_warning,
            'warning_msg': warning_msg
        })
        
        print(f"✓ Epoch {epoch} Grad-CAM分析完成: {save_path}")
        if has_warning:
            print(f"⚠️ 警告: {warning_msg}")
        else:
            print(f"✓ 水印关注度检查通过")
        
        return analysis_result
    
    def _add_region_markers(self, ax, w, h):
        """在图像上添加区域标记"""
        # 水印区域标记（红色）
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = self.watermark_regions['medium']
        x1, y1 = int(x1_ratio * w), int(y1_ratio * h)
        x2, y2 = int(x2_ratio * w), int(y2_ratio * h)
        
        watermark_rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
        ax.add_patch(watermark_rect)
        watermark_label = '水印区域' if self.font_available else 'Watermark'
        ax.text(x1, y1-5, watermark_label, color='red', fontsize=10, weight='bold')
        
        # 中心区域标记（绿色）
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = self.center_region
        x1, y1 = int(x1_ratio * w), int(y1_ratio * h)
        x2, y2 = int(x2_ratio * w), int(y2_ratio * h)
        
        center_rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=2, edgecolor='green', facecolor='none', alpha=0.8)
        ax.add_patch(center_rect)
        center_label = '中心区域' if self.font_available else 'Center Region'
        ax.text(x1, y1-5, center_label, color='green', fontsize=10, weight='bold')
    
    def _format_analysis_text(self, analysis_result, has_warning, warning_msg):
        """格式化分析结果文本"""
        regions_attention = analysis_result['regions_attention']
        attention_ratios = analysis_result['attention_ratios']
        
        text = "注意力区域分析:\n\n" if self.font_available else "Attention Analysis:\n\n"
        
        # 区域注意力值
        text += "区域注意力强度:\n" if self.font_available else "Region Attention:\n"
        text += f"中心区域: {regions_attention['center']:.3f}\n"
        text += f"水印区域(小): {regions_attention['watermark_small']:.3f}\n"
        text += f"水印区域(中): {regions_attention['watermark_medium']:.3f}\n"
        text += f"水印区域(大): {regions_attention['watermark_large']:.3f}\n\n"
        
        # 注意力比例
        text += "水印/中心注意力比例:\n" if self.font_available else "Watermark/Center Ratios:\n"
        text += f"小区域比例: {attention_ratios['small_ratio']:.3f}\n"
        text += f"中区域比例: {attention_ratios['medium_ratio']:.3f}\n"
        text += f"大区域比例: {attention_ratios['large_ratio']:.3f}\n\n"
        
        # 警告状态
        status = "⚠️ 警告" if has_warning else "✓ 正常"
        text += f"状态: {status}\n"
        text += f"详情: {warning_msg}\n\n"
        
        # 建议
        if has_warning:
            text += "建议:\n" if self.font_available else "Suggestions:\n"
            text += "• 检查数据增强策略\n"
            text += "• 考虑添加注意力正则化\n"
            text += "• 监控后续训练过程\n"
        else:
            text += "模型关注区域正确" if self.font_available else "Model attention is correct"
        
        return text
    
    def generate_monitoring_report(self, save_name='training_monitoring_report'):
        """生成训练监控报告"""
        if not self.attention_history:
            print("⚠️ 没有监控数据，跳过报告生成")
            return
        
        # 提取数据
        epochs = [item['epoch'] for item in self.attention_history]
        center_attentions = [item['analysis']['regions_attention']['center'] for item in self.attention_history]
        watermark_ratios = [item['analysis']['attention_ratios']['medium_ratio'] for item in self.attention_history]
        warnings = [item['has_warning'] for item in self.attention_history]
        
        # 创建报告图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练监控报告' if self.font_available else 'Training Monitoring Report', fontsize=16)
        
        # 中心区域注意力趋势
        axes[0, 0].plot(epochs, center_attentions, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_title('中心区域注意力趋势' if self.font_available else 'Center Region Attention Trend')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('注意力强度' if self.font_available else 'Attention Intensity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 水印/中心注意力比例趋势
        axes[0, 1].plot(epochs, watermark_ratios, 'r-o', linewidth=2, markersize=4)
        axes[0, 1].axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='警告阈值')
        axes[0, 1].set_title('水印/中心注意力比例' if self.font_available else 'Watermark/Center Ratio')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('比例' if self.font_available else 'Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 警告统计
        warning_epochs = [epoch for epoch, warning in zip(epochs, warnings) if warning]
        axes[1, 0].bar(['正常', '警告'] if self.font_available else ['Normal', 'Warning'], 
                      [len(epochs) - len(warning_epochs), len(warning_epochs)],
                      color=['green', 'red'], alpha=0.7)
        axes[1, 0].set_title('警告统计' if self.font_available else 'Warning Statistics')
        axes[1, 0].set_ylabel('Epoch数量' if self.font_available else 'Number of Epochs')
        
        # 监控总结
        summary_text = self._generate_summary_text(epochs, center_attentions, watermark_ratios, warnings)
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        # 保存报告
        report_path = os.path.join(self.save_dir, f'{save_name}.png')
        plt.tight_layout()
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 训练监控报告已保存: {report_path}")
        return report_path
    
    def _generate_summary_text(self, epochs, center_attentions, watermark_ratios, warnings):
        """生成监控总结文本"""
        total_epochs = len(epochs)
        warning_count = sum(warnings)
        avg_center_attention = np.mean(center_attentions)
        avg_watermark_ratio = np.mean(watermark_ratios)
        max_watermark_ratio = np.max(watermark_ratios)
        
        text = "监控总结:\n\n" if self.font_available else "Monitoring Summary:\n\n"
        text += f"总监控轮数: {total_epochs}\n"
        text += f"警告次数: {warning_count}\n"
        text += f"警告率: {warning_count/total_epochs*100:.1f}%\n\n"
        
        text += f"平均中心注意力: {avg_center_attention:.3f}\n"
        text += f"平均水印比例: {avg_watermark_ratio:.3f}\n"
        text += f"最大水印比例: {max_watermark_ratio:.3f}\n\n"
        
        # 评估结果
        if warning_count == 0:
            text += "✓ 训练过程正常\n"
            text += "模型正确关注中心区域"
        elif warning_count / total_epochs < 0.3:
            text += "⚠️ 偶有异常\n"
            text += "建议继续监控"
        else:
            text += "❌ 频繁警告\n"
            text += "建议调整训练策略"
        
        return text


class GradCAM:
    """Grad-CAM实现"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # 使用新的hook方式避免警告
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __call__(self, input_tensor, target_index=None):
        # 确保模型处于训练模式以计算梯度
        self.model.train()
        
        # 确保输入tensor需要梯度
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.requires_grad_(True)
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 处理不同模型的输出格式
        if isinstance(output, tuple):
            # CNN模型返回(concentration, features)元组
            target_output = output[0]
        else:
            # VGG模型直接返回concentration
            target_output = output
        
        # 如果是回归任务，使用输出值作为目标
        if target_index is None:
            target = target_output.mean()
        else:
            target = target_output[0, target_index]
        
        # 确保target需要梯度
        if not target.requires_grad:
            print("⚠️ 警告：target不需要梯度，GradCAM可能失败")
            return np.zeros((224, 224))
        
        # 反向传播
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # 检查梯度和激活是否存在
        if self.gradients is None or self.activations is None:
            print("⚠️ 警告：梯度或激活为空，返回零矩阵")
            return np.zeros((224, 224))
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 生成CAM
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # 应用ReLU并归一化
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # 确保输出尺寸为224x224
        cam_np = cam.detach().cpu().numpy()
        if cam_np.shape != (224, 224):
            import cv2
            cam_np = cv2.resize(cam_np, (224, 224))
        
        return cam_np 