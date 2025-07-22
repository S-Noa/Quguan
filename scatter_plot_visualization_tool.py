#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
散点图公式化工具
支持云端ResNet50、VGG模型和本地基线CNN模型的预测值vs真实值可视化
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
from datetime import datetime
from pathlib import Path
import seaborn as sns
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict, Any, Union

# 添加路径
sys.path.append('src')
sys.path.append('cloud_vgg_training_package')

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def safe_collate_fn(batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    安全的collate函数，处理可能缺失的字段
    """
    try:
        # 分离图像、浓度和元数据
        images = []
        concentrations = []
        metadata = []
        
        for item in batch:
            if len(item) == 3:
                image, concentration, meta = item
                images.append(image)
                concentrations.append(concentration)
                
                # 确保元数据包含必要字段
                if isinstance(meta, dict):
                    # 添加缺失的字段
                    if 'detection_bbox' not in meta:
                        meta['detection_bbox'] = [0, 0, 224, 224]  # 默认整个图像
                    if 'detection_confidence' not in meta:
                        meta['detection_confidence'] = 1.0
                    metadata.append(meta)
                else:
                    # 如果meta不是字典，创建一个默认的
                    metadata.append({
                        'detection_bbox': [0, 0, 224, 224],
                        'detection_confidence': 1.0,
                        'bg_type': 'unknown',
                        'power': 'unknown',
                        'distance': 'unknown'
                    })
            else:
                raise ValueError(f"Unexpected item format: {len(item)} elements")
        
        # 使用默认的collate函数处理
        from torch.utils.data.dataloader import default_collate
        images_tensor = default_collate(images)
        concentrations_tensor = default_collate(concentrations)
        
        return images_tensor, concentrations_tensor, metadata
        
    except Exception as e:
        print(f"Collate函数错误: {e}")
        print(f"批次大小: {len(batch)}")
        if batch:
            print(f"第一个项目类型: {type(batch[0])}")
            print(f"第一个项目长度: {len(batch[0]) if hasattr(batch[0], '__len__') else 'N/A'}")
        raise

class ScatterPlotVisualizer:
    """散点图可视化器"""
    
    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录，默认为当前目录下的scatter_plot_analysis_时间戳
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"scatter_plot_analysis_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"📊 输出目录: {self.output_dir}")
        print(f"🔧 使用设备: {self.device}")
    
    def _load_checkpoint_safely(self, model_path):
        """安全加载checkpoint，兼容PyTorch 2.6"""
        try:
            # 首先尝试使用weights_only=True（安全模式）
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception as e:
            # 如果安全模式失败，使用weights_only=False（兼容旧格式）
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint
    
    def load_model_and_predict(self, model_type: str, model_path: str, dataset_path: str, bg_mode: str = 'all') -> Tuple[np.ndarray, np.ndarray, str]:
        """
        加载模型并生成预测
        
        Args:
            model_type: 模型类型 ('baseline_cnn', 'enhanced_cnn', 'cloud_resnet50', 'cloud_vgg', 'adaptive_resnet50')
            model_path: 模型路径
            dataset_path: 数据集路径
            bg_mode: 背景模式 ('all', 'bg0', 'bg1', 'bg0_20mw', 'bg0_100mw', 'bg0_400mw', 'bg1_20mw', 'bg1_100mw', 'bg1_400mw')
        """
        if model_type == 'baseline_cnn':
            return self._load_baseline_cnn(model_path, dataset_path, bg_mode)
        elif model_type == 'enhanced_cnn':
            return self._load_enhanced_cnn(model_path, dataset_path, bg_mode)
        elif model_type == 'cloud_resnet50':
            return self._load_cloud_resnet50(model_path, dataset_path, bg_mode)
        elif model_type == 'cloud_vgg':
            return self._load_cloud_vgg(model_path, dataset_path, bg_mode)
        elif model_type == 'adaptive_resnet50':
            return self._load_adaptive_resnet50(model_path, dataset_path, bg_mode)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _load_baseline_cnn(self, model_path: str, dataset_path: str, bg_mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """加载基线CNN模型"""
        from cnn_model import CNNFeatureExtractor
        from feature_dataset_loader import create_feature_dataloader
        
        # 加载模型
        model = CNNFeatureExtractor()
        checkpoint = self._load_checkpoint_safely(model_path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # 加载数据
        bg_filter, power_filter = self._parse_bg_mode(bg_mode)
        _, dataset = create_feature_dataloader(
            feature_dataset_path=dataset_path,
            batch_size=32,
            shuffle=False,
            bg_type=bg_filter,
            power_filter=power_filter
        )
        
        # 手动创建DataLoader使用safe_collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=safe_collate_fn
        )
        
        # 生成预测
        predictions, targets = self._generate_predictions(model, dataloader)
        
        return predictions, targets, "基线CNN"
    
    def _load_enhanced_cnn(self, model_path: str, dataset_path: str, bg_mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """加载增强CNN模型"""
        from enhanced_laser_spot_cnn import EnhancedLaserSpotCNN
        from feature_dataset_loader import create_feature_dataloader
        
        # 加载模型
        model = EnhancedLaserSpotCNN(num_classes=1)
        checkpoint = self._load_checkpoint_safely(model_path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # 加载数据
        bg_filter, power_filter = self._parse_bg_mode(bg_mode)
        _, dataset = create_feature_dataloader(
            feature_dataset_path=dataset_path,
            batch_size=32,
            shuffle=False,
            bg_type=bg_filter,
            power_filter=power_filter
        )
        
        # 手动创建DataLoader使用safe_collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=safe_collate_fn
        )
        
        # 生成预测
        predictions, targets = self._generate_predictions(model, dataloader)
        
        return predictions, targets, "增强CNN"
    
    def _load_cloud_resnet50(self, model_path: str, dataset_path: str, bg_mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """加载云端ResNet50模型"""
        from feature_dataset_loader import create_feature_dataloader
        
        # 尝试导入云端ResNet50回归模型
        try:
            import sys
            sys.path.append('cloud_vgg_training_package/src')
            from resnet_regression import ResNet50Regression
            
            # 创建模型（与训练时相同的架构）
            model = ResNet50Regression(freeze_backbone=False, dropout_rate=0.5)
            print("✅ 使用云端ResNet50回归模型架构")
            
        except ImportError:
            print("⚠️  无法导入云端ResNet50模型，尝试标准ResNet50架构")
            import torchvision.models as models
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 1)
        
        # 加载权重
        checkpoint = self._load_checkpoint_safely(model_path)
        
        # 智能加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 尝试加载权重，使用strict=False处理可能的架构不匹配
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ 严格模式加载权重成功")
        except RuntimeError as e:
            print(f"⚠️  严格模式失败: {str(e)[:200]}...")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("✅ 宽松模式加载权重成功")
            except RuntimeError as e2:
                print(f"❌ 权重加载失败: {e2}")
                raise e2
        
        model.to(self.device)
        model.eval()

        # 加载数据
        bg_filter, power_filter = self._parse_bg_mode(bg_mode)
        _, dataset = create_feature_dataloader(
            feature_dataset_path=dataset_path,
            batch_size=32,
            shuffle=False,
            bg_type=bg_filter,
            power_filter=power_filter
        )
        
        # 手动创建DataLoader使用safe_collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=safe_collate_fn
        )
        
        # 生成预测
        predictions, targets = self._generate_predictions(model, dataloader)
        
        return predictions, targets, "云端ResNet50"
    
    def _load_cloud_vgg(self, model_path: str, dataset_path: str, bg_mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """加载云端VGG模型"""
        import sys
        sys.path.append('cloud_vgg_training_package/src')
        from feature_dataset_loader import create_feature_dataloader
        from vgg_regression import VGGRegressionCBAM
        
        # 使用正确的VGG+CBAM模型架构
        model = VGGRegressionCBAM(
            freeze_features=False,
            debug_mode=False
        )
        
        checkpoint = self._load_checkpoint_safely(model_path)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # 加载数据
        bg_filter, power_filter = self._parse_bg_mode(bg_mode)
        _, dataset = create_feature_dataloader(
            feature_dataset_path=dataset_path,
            batch_size=32,
            shuffle=False,
            bg_type=bg_filter,
            power_filter=power_filter
        )
        
        # 手动创建DataLoader使用safe_collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=safe_collate_fn
        )
        
        # 生成预测
        predictions, targets = self._generate_predictions(model, dataloader)
        
        return predictions, targets, "云端VGG"
    
    def _load_adaptive_resnet50(self, model_path: str, dataset_path: str, bg_mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """加载自适应ResNet50模型"""
        from adaptive_attention_resnet50 import AdaptiveAttentionResNet50
        from feature_dataset_loader import create_feature_dataloader
        
        # 加载模型
        model = AdaptiveAttentionResNet50(num_classes=1)
        checkpoint = self._load_checkpoint_safely(model_path)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # 加载数据
        bg_filter, power_filter = self._parse_bg_mode(bg_mode)
        _, dataset = create_feature_dataloader(
            feature_dataset_path=dataset_path,
            batch_size=32,
            shuffle=False,
            bg_type=bg_filter,
            power_filter=power_filter
        )
        
        # 手动创建DataLoader使用safe_collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=safe_collate_fn
        )
        
        # 生成预测
        predictions, targets = self._generate_predictions(model, dataloader)
        
        return predictions, targets, "自适应ResNet50"
    
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
    
    def _generate_predictions(self, model, dataloader):
        """生成预测值"""
        predictions = []
        targets = []
        
        print(f"   正在生成预测... (共{len(dataloader)}个批次)")
        
        with torch.no_grad():
            for batch_idx, (images, batch_targets, metadata) in enumerate(dataloader):
                images = images.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = model(images)
                
                # 处理可能的tuple输出
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                predictions.extend(outputs.squeeze().cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"   进度: {batch_idx + 1}/{len(dataloader)}")
        
        return np.array(predictions), np.array(targets)
    
    def create_scatter_plot(self, predictions: np.ndarray, targets: np.ndarray, model_name: str, save_name: Optional[str] = None, min_concentration: Optional[float] = None, max_concentration: Optional[float] = None) -> Tuple[float, float, float]:
        """创建散点图"""
        if save_name is None:
            save_name = f"scatter_plot_{model_name.replace(' ', '_').replace('+', '_')}"

        # 应用浓度范围筛选
        mask: np.ndarray[np.bool_] = np.ones_like(targets, dtype=bool)
        if min_concentration is not None:
            mask &= (targets >= min_concentration)
        if max_concentration is not None:
            mask &= (targets <= max_concentration)

        # 应用筛选
        filtered_predictions: np.ndarray = predictions[mask]
        filtered_targets: np.ndarray = targets[mask]

        # 如果所有数据都被筛选掉，发出警告并使用原始数据
        if len(filtered_targets) == 0:
            print(f"⚠️ 警告: 浓度范围 [{min_concentration}, {max_concentration}] mg/L 内没有数据点，将使用所有数据")
            filtered_predictions = predictions
            filtered_targets = targets
        elif len(filtered_targets) < len(targets):
            print(f"🔍 已筛选浓度范围: [{min_concentration or min(targets):.2f}, {max_concentration or max(targets):.2f}] mg/L")
            print(f"   原始数据点: {len(targets)}, 筛选后数据点: {len(filtered_targets)}")

        # 计算评估指标
        r2 = r2_score(filtered_targets, filtered_predictions)
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        # 计算误差
        errors: np.ndarray = filtered_predictions - filtered_targets

        print(f"\n📊 {model_name} 评估指标 (筛选后):")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   样本数: {len(targets)}")
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 主散点图
        ax1.scatter(filtered_targets, filtered_predictions, alpha=0.6, s=20)

        # 理想预测线
        min_val = min(min(filtered_targets), min(filtered_predictions))
        max_val = max(max(filtered_targets), max(filtered_predictions))

        # 添加浓度范围标注
        if min_concentration is not None or max_concentration is not None:
            range_text = f"浓度范围: {min_concentration if min_concentration is not None else 'min'} - {max_concentration if max_concentration is not None else 'max'} mg/L"
            ax1.text(0.05, 0.90, range_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测')
        
        ax1.set_xlabel('真实浓度 (mg/L)')
        ax1.set_ylabel('预测浓度 (mg/L)')
        ax1.set_title(f'{model_name} - 预测 vs 真实值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加评估指标文本
        textstr = f'R² = {r2:.4f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\nN = {len(targets)}'
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. 误差分布
        ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差')
        ax2.set_xlabel('预测误差 (mg/L)')
        ax2.set_ylabel('频次')
        ax2.set_title('预测误差分布 (筛选后)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差图
        ax3.scatter(filtered_targets, errors, alpha=0.6, s=20)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('真实浓度 (mg/L)')
        ax3.set_ylabel('残差 (mg/L)')
        ax3.set_title('残差图 (筛选后)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 浓度分布对比
        ax4.hist(filtered_targets, bins=30, alpha=0.7, label='真实值', density=True)
        ax4.hist(filtered_predictions, bins=30, alpha=0.7, label='预测值', density=True)
        ax4.set_xlabel('浓度 (mg/L)')
        ax4.set_ylabel('密度')
        ax4.set_title('浓度分布对比 (筛选后)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   散点图已保存: {save_path}")
        
        # 保存数据
        data_path = os.path.join(self.output_dir, f"{save_name}_data.json")
        data = {
            'model_name': model_name,
            'metrics': {
                'r2_score': float(r2),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse)
            },
            'sample_count': len(targets),
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'errors': errors.tolist()
        }
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"   数据已保存: {data_path}")
        
        return r2, mae, rmse
    
    def create_comparison_plot(self, model_results: List[Tuple[np.ndarray, np.ndarray, str]]) -> None:
        """创建模型对比图"""
        if len(model_results) < 2:
            print("⚠️ 需要至少2个模型才能创建对比图")
            return
        
        print(f"\n📈 创建模型对比图...")
        
        # 创建对比图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # 1. 所有模型的散点图
        for i, (predictions, targets, model_name) in enumerate(model_results):
            color = colors[i % len(colors)]
            ax1.scatter(targets, predictions, alpha=0.6, s=15, 
                       color=color, label=model_name)
        
        # 理想预测线
        all_targets = np.concatenate([targets for _, targets, _ in model_results])
        all_predictions = np.concatenate([predictions for predictions, _, _ in model_results])
        min_val = min(min(all_targets), min(all_predictions))
        max_val = max(max(all_targets), max(all_predictions))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='理想预测')
        
        ax1.set_xlabel('真实浓度 (mg/L)')
        ax1.set_ylabel('预测浓度 (mg/L)')
        ax1.set_title('模型对比 - 预测 vs 真实值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 评估指标对比
        model_names = []
        r2_scores = []
        mae_scores = []
        rmse_scores = []
        
        for predictions, targets, model_name in model_results:
            model_names.append(model_name)
            r2_scores.append(r2_score(targets, predictions))
            mae_scores.append(mean_absolute_error(targets, predictions))
            rmse_scores.append(np.sqrt(mean_squared_error(targets, predictions)))
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax2.bar(x - width, r2_scores, width, label='R² Score', alpha=0.8)
        ax2.set_xlabel('模型')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score 对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. MAE对比
        ax3.bar(x, mae_scores, width, label='MAE', alpha=0.8, color='orange')
        ax3.set_xlabel('模型')
        ax3.set_ylabel('MAE (mg/L)')
        ax3.set_title('平均绝对误差对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. RMSE对比
        ax4.bar(x, rmse_scores, width, label='RMSE', alpha=0.8, color='red')
        ax4.set_xlabel('模型')
        ax4.set_ylabel('RMSE (mg/L)')
        ax4.set_title('均方根误差对比')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_path = os.path.join(self.output_dir, "model_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   模型对比图已保存: {comparison_path}")
        
        # 保存对比数据
        comparison_data = {
            'comparison_summary': {
                'model_names': model_names,
                'r2_scores': r2_scores,
                'mae_scores': mae_scores,
                'rmse_scores': rmse_scores
            },
            'best_models': {
                'highest_r2': model_names[np.argmax(r2_scores)],
                'lowest_mae': model_names[np.argmin(mae_scores)],
                'lowest_rmse': model_names[np.argmin(rmse_scores)]
            }
        }
        
        comparison_data_path = os.path.join(self.output_dir, "model_comparison_data.json")
        with open(comparison_data_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print(f"   对比数据已保存: {comparison_data_path}")
        
        # 打印最佳模型
        print(f"\n🏆 最佳模型:")
        print(f"   最高R²: {comparison_data['best_models']['highest_r2']} (R²={max(r2_scores):.4f})")
        print(f"   最低MAE: {comparison_data['best_models']['lowest_mae']} (MAE={min(mae_scores):.2f})")
        print(f"   最低RMSE: {comparison_data['best_models']['lowest_rmse']} (RMSE={min(rmse_scores):.2f})")

def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description='散点图公式化工具')
    
    # 基本参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='特征数据集路径')
    parser.add_argument('--bg_mode', type=str, default='all',
                       help='背景模式 (bg0, bg1, all, bg0_20mw, etc.)')

    # 浓度范围筛选
    parser.add_argument('--min_concentration', type=float, default=None,
                       help='最小浓度值 (mg/L)')
    parser.add_argument('--max_concentration', type=float, default=None,
                       help='最大浓度值 (mg/L)')

    # 模型配置
    parser.add_argument('--baseline_cnn_model', type=str, default=None,
                       help='基线CNN模型路径')
    parser.add_argument('--enhanced_cnn_model', type=str, default=None,
                       help='增强CNN模型路径')
    parser.add_argument('--cloud_resnet50_model', type=str, default=None,
                       help='云端ResNet50模型路径')
    parser.add_argument('--cloud_vgg_model', type=str, default=None,
                       help='云端VGG模型路径')
    parser.add_argument('--adaptive_resnet50_model', type=str, default=None,
                       help='自适应ResNet50模型路径')
    
    # 可视化选项
    parser.add_argument('--create_comparison', action='store_true',
                       help='创建模型对比图')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['baseline_cnn', 'enhanced_cnn', 'cloud_resnet50', 'cloud_vgg', 'adaptive_resnet50'],
                       help='要分析的模型类型')
    
    args = parser.parse_args()
    
    print("📊 散点图公式化工具")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = ScatterPlotVisualizer(args.output_dir)
    
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
    
    # 收集要分析的模型
    models_to_analyze = []
    model_results = []
    
    if args.models:
        # 使用指定的模型列表
        for model_type in args.models:
            model_path = getattr(args, f"{model_type}_model")
            if model_path and os.path.exists(model_path):
                models_to_analyze.append((model_type, model_path))
            else:
                print(f"⚠️ {model_type}模型路径未指定或文件不存在: {model_path}")
    else:
        # 自动检测所有可用模型
        model_configs = [
            ('baseline_cnn', args.baseline_cnn_model),
            ('enhanced_cnn', args.enhanced_cnn_model),
            ('cloud_resnet50', args.cloud_resnet50_model),
            ('cloud_vgg', args.cloud_vgg_model),
            ('adaptive_resnet50', args.adaptive_resnet50_model)
        ]
        
        for model_type, model_path in model_configs:
            if model_path and os.path.exists(model_path):
                models_to_analyze.append((model_type, model_path))
    
    if not models_to_analyze:
        print("❌ 未找到可用的模型文件")
        return
    
    print(f"📋 将分析以下模型:")
    for model_type, model_path in models_to_analyze:
        print(f"   {model_type}: {model_path}")
    
    # 分析每个模型
    for model_type, model_path in models_to_analyze:
        try:
            print(f"\n{'='*60}")
            print(f"🔍 分析模型: {model_type}")
            
            predictions, targets, model_name = visualizer.load_model_and_predict(
                model_type, model_path, args.dataset_path, args.bg_mode
            )
            
            r2, mae, rmse = visualizer.create_scatter_plot(
                predictions, targets, model_name,
                min_concentration=args.min_concentration,
                max_concentration=args.max_concentration
            )
            
            model_results.append((predictions, targets, model_name))
            
        except Exception as e:
            print(f"❌ 分析{model_type}模型失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 创建模型对比图
    if args.create_comparison and len(model_results) > 1:
        visualizer.create_comparison_plot(model_results)
    
    print(f"\n🎉 散点图分析完成!")
    print(f"   结果保存在: {visualizer.output_dir}")

if __name__ == "__main__":
    main()