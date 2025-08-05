#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用特征数据集训练回归模型
支持CNN、VGG等模型，使用YOLO裁剪的特征区域进行训练
"""

import os
import sys
import argparse
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import logging

# 添加src目录到路径
sys.path.append('src')

# 添加数据集名称工具
try:
    from dataset_name_utils import generate_training_output_dir, parse_training_mode_from_args, get_dataset_info_string
    DATASET_UTILS_AVAILABLE = True
    print("数据集名称工具加载成功")
except ImportError:
    print("警告：无法导入数据集名称工具，将使用传统命名方式")
    DATASET_UTILS_AVAILABLE = False

from feature_dataset_loader import create_feature_dataloader, detect_feature_datasets
from advanced_cnn_models import AdvancedLaserSpotCNN, create_advanced_model
from vgg_regression import VGGRegressionCBAM
from resnet_regression import ResNet50Regression

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=15, min_delta=1.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, current_loss, model):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(model)
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}


def safe_collate_fn(batch):
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
            print(f"第一个元素类型: {type(batch[0])}")
            if len(batch[0]) > 0:
                print(f"第一个元素内容: {batch[0]}")
        raise


def setup_logger(log_file=None, group_tag=None):
    """设置日志记录"""
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f'feature_training_{group_tag}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('Feature_Training')
    return logger, log_file

class SmoothAdaptiveWeightedLoss(nn.Module):
    """
    平滑自适应权重损失函数
    解决硬阈值问题，使用Sigmoid实现平滑过渡
    """
    
    def __init__(self, transition_concentration=800, beta=0.5, scale_factor=0.01):
        super(SmoothAdaptiveWeightedLoss, self).__init__()
        self.transition_concentration = transition_concentration
        self.beta = beta
        self.scale_factor = scale_factor
        print(f"🎯 初始化平滑自适应权重损失函数:")
        print(f"   过渡浓度: {transition_concentration} mg/L")
        print(f"   最小权重: {beta}")
        print(f"   平滑因子: {scale_factor}")
        
    def forward(self, predictions, targets):
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # 计算基础MSE误差
        squared_errors = (predictions - targets) ** 2
        
        # 计算平滑权重
        scale = self.scale_factor * self.transition_concentration
        weights = self.beta + (1 - self.beta) * torch.sigmoid(
            scale * (self.transition_concentration - targets)
        )
        
        # 加权平均
        return torch.mean(weights * squared_errors)

class ProgressiveSmoothWeightedLoss(SmoothAdaptiveWeightedLoss):
    """
    渐进式平滑权重损失函数
    从标准MSE逐渐过渡到平滑权重
    """
    
    def __init__(self, transition_concentration=800, final_beta=0.5, 
                 warmup_epochs=30, scale_factor=0.01):
        super().__init__(transition_concentration, final_beta, scale_factor)
        self.final_beta = final_beta
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.current_beta = 1.0
        print(f"🎯 初始化渐进式平滑权重损失函数:")
        print(f"   过渡浓度: {transition_concentration} mg/L")
        print(f"   最终权重: {final_beta}")
        print(f"   预热轮次: {warmup_epochs}")
        print(f"   平滑因子: {scale_factor}")
        
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            self.current_beta = 1.0 - progress * (1.0 - self.final_beta)
        else:
            self.current_beta = self.final_beta
        self.beta = self.current_beta

class ConcentrationAwareLoss(nn.Module):
    """
    浓度感知损失函数 - 传统硬阈值版本
    保留作为兼容选项
    """
    
    def __init__(self, high_concentration_threshold=800, high_concentration_weight=0.5):
        super(ConcentrationAwareLoss, self).__init__()
        self.threshold = high_concentration_threshold
        self.high_weight = high_concentration_weight
        print(f"🎯 初始化浓度感知损失函数:")
        print(f"   高浓度阈值: {high_concentration_threshold} mg/L")
        print(f"   高浓度损失权重: {high_concentration_weight}")
        
    def forward(self, predictions, targets):
        """
        前向传播
        Args:
            predictions: 模型预测值 [batch_size] 
            targets: 真实浓度值 [batch_size]
        """
        # 确保输入为1D张量
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # 分离低浓度和高浓度样本
        low_mask = targets < self.threshold
        high_mask = targets >= self.threshold
        
        total_loss = 0.0
        loss_components = {}
        
        # 低浓度:使用标准MSE损失
        if low_mask.sum() > 0:
            low_preds = predictions[low_mask]
            low_targets = targets[low_mask]
            low_loss = torch.nn.functional.mse_loss(low_preds, low_targets)
            total_loss += low_loss
            loss_components['low_concentration'] = low_loss.item()
        else:
            loss_components['low_concentration'] = 0.0
        
        # 高浓度:使用相对MSE损失(降权处理)
        if high_mask.sum() > 0:
            high_preds = predictions[high_mask]
            high_targets = targets[high_mask]
            
            # 计算相对误差,避免除零
            relative_errors = (high_preds - high_targets) / torch.clamp(high_targets, min=100.0)
            high_loss = torch.mean(relative_errors ** 2) * self.high_weight
            
            total_loss += high_loss
            loss_components['high_concentration'] = high_loss.item()
        else:
            loss_components['high_concentration'] = 0.0
        
        return total_loss

class FeatureDatasetTrainer:
    """特征数据集训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置输出目录，优先使用数据集名称工具
        try:
            # 导入数据集名称工具
            from dataset_name_utils import generate_training_output_dir, parse_training_mode_from_args
            
            # 解析训练模式
            bg_filter, power_filter, training_mode = parse_training_mode_from_args(args)
            
            # 添加冻结状态标识
            freeze_suffix = self._get_freeze_suffix(args)
            model_name_with_suffix = f"{args.model_type}{freeze_suffix}"
            
            # 生成基于数据集的输出目录
            dataset_path = getattr(args, 'feature_dataset_path', None)
            if dataset_path:
                self.output_dir = generate_training_output_dir(
                    model_name=model_name_with_suffix,
                    dataset_path=dataset_path,
                    training_mode=training_mode,
                    bg_filter=bg_filter,
                    power_filter=power_filter
                )
            else:
                # 如果没有数据集路径，使用传统方式
                raise ImportError("数据集路径未指定")
                
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"使用基于数据集的输出目录: {self.output_dir}")
            
        except (ImportError, Exception) as e:
            # 回退到传统方式
            print(f"生成基于数据集的输出目录失败: {e}，使用传统方式")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_suffix = args.bg_mode.replace('_', '-')
            freeze_suffix = self._get_freeze_suffix(args)
            self.output_dir = f"feature_training_{args.model_type}_{mode_suffix}{freeze_suffix}_results_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self.logger, _ = setup_logger(
            os.path.join(self.output_dir, 'training.log'), 
            f'{args.model_type}_{args.bg_mode.replace("_", "-")}'
        )
        
        self.logger.info(f"{args.model_type.upper()}训练器初始化")
        self.logger.info(f"   设备: {self.device}")
        self.logger.info(f"   模型类型: {args.model_type}")
        self.logger.info(f"   训练模式: {args.bg_mode.upper()}")
        self.logger.info(f"   输出目录: {self.output_dir}")
        
        # 添加浓度范围过滤信息到日志
        if args.concentration_range:
            self.logger.info(f"   浓度范围过滤: {args.concentration_range}")
        else:
            self.logger.info("   浓度范围过滤: 未启用")
        
        # 添加数据集信息到日志
        try:
            from dataset_name_utils import get_dataset_info_string
            dataset_path = getattr(args, 'feature_dataset_path', None)
            if dataset_path:
                bg_filter, power_filter, _ = parse_training_mode_from_args(args)
                dataset_info = get_dataset_info_string(dataset_path, bg_filter, power_filter)
                self.logger.info(f"   {dataset_info}")
        except:
            pass
    
    def _detect_parameter_changes(self):
        """智能检测是否需要重置优化器/调度器参数"""
        self.logger.info("=== 智能参数检测 ===")
        
        if not self.args.resume:
            self.logger.info("   非断点续训模式，跳过参数检测")
            return
        
        # 检查是否显式指定了重置参数
        if self.args.reset_optimizer_on_resume or self.args.reset_scheduler_on_resume:
            self.logger.info("   已显式指定参数重置标志，跳过智能检测")
            return
        
        # 智能检测：如果指定了关键训练参数，自动启用重置
        parser = argparse.ArgumentParser()
        
        # 添加默认参数（复制main函数中的参数定义）
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        parser.add_argument('--scheduler', type=str, default='step')
        parser.add_argument('--step_size', type=int, default=50)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument('--patience', type=int, default=10)
        parser.add_argument('--weight_decay', type=float, default=1e-3)
        parser.add_argument('--optimizer', type=str, default='adam')
        
        # 解析默认值
        defaults = parser.parse_args([])
        
        # 检查关键参数是否被修改
        changed_params = []
        critical_params = ['learning_rate', 'scheduler', 'step_size', 'gamma', 'patience', 'weight_decay', 'optimizer']
        
        self.logger.info("   检查关键参数变化:")
        for param in critical_params:
            if hasattr(self.args, param) and hasattr(defaults, param):
                current_val = getattr(self.args, param)
                default_val = getattr(defaults, param)
                if current_val != default_val:
                    changed_params.append(param)
                    self.logger.info(f"     ✓ {param}: {default_val} -> {current_val}")
                else:
                    self.logger.info(f"     - {param}: {current_val} (默认值)")
        
        # 如果有关键参数被修改，自动启用重置
        if changed_params:
            self.logger.info(f"🔍 检测到训练参数变更: {', '.join(changed_params)}")
            self.logger.info("💡 自动启用参数重置模式（保留模型权重，重置优化器和调度器）")
            self.args.reset_optimizer_on_resume = True
            self.args.reset_scheduler_on_resume = True
            self._parameter_changes_detected = changed_params
        else:
            self.logger.info("🔄 使用检查点原始参数继续训练")
            self._parameter_changes_detected = []
    
    def _get_freeze_suffix(self, args):
        """生成冻结状态后缀，用于区分输出目录"""
        freeze_parts = []
        
        if args.model_type == 'vgg':
            if args.freeze_features:
                freeze_parts.append('frozen')
            else:
                freeze_parts.append('unfrozen')
        elif args.model_type == 'resnet50':
            if args.freeze_backbone:
                freeze_parts.append('frozen')
            else:
                freeze_parts.append('unfrozen')
        elif args.model_type == 'cnn':
            # CNN模型没有预训练权重，不需要冻结标识
            pass
        
        # 添加渐进式解冻标识
        if args.progressive_unfreeze:
            freeze_parts.append('progressive')
        
        if freeze_parts:
            return '_' + '_'.join(freeze_parts)
        else:
            return ''
    
    def _parse_training_mode(self):
        """解析训练模式，支持6档细分"""
        bg_mode = self.args.bg_mode
        
        # 6档细分模式
        if '_' in bg_mode:
            # 解析 bg0_20mw, bg1_100mw 等格式
            parts = bg_mode.split('_')
            if len(parts) == 2:
                bg_filter = parts[0]  # bg0 或 bg1
                power_filter = parts[1]  # 20mw, 100mw, 400mw
                
                self.logger.info(f">>> 6档细分训练模式: {bg_mode}")
                self.logger.info(f"   光照条件: {bg_filter}")
                self.logger.info(f"   激光功率: {power_filter}")
                
                # 忽略命令行的power_filter参数
                if self.args.power_filter:
                    self.logger.warning(f"   !!! 忽略命令行power_filter: {self.args.power_filter}")
                
                return bg_filter, power_filter
            else:
                raise ValueError(f"无效的细分模式格式: {bg_mode}")
        
        # 传统3档模式
        else:
            power_filter = self.args.power_filter  # 可能为None
            
            if bg_mode == 'all':
                bg_filter = None
                self.logger.info(f">>> 传统训练模式: 全部数据")
            else:
                bg_filter = bg_mode  # bg0 或 bg1
                self.logger.info(f">>> 传统训练模式: {bg_filter}")
            
            if power_filter:
                self.logger.info(f"   额外功率过滤: {power_filter}")
            
            return bg_filter, power_filter
    
    def load_data(self):
        """加载特征数据集"""
        self.logger.info("=== 第1阶段：加载特征数据集 ===")
        
        # 自动检测特征数据集（支持版本选择）
        if self.args.feature_dataset_path is None:
            self.logger.info(f"自动检测特征数据集 (版本: {self.args.dataset_version})...")
            feature_datasets = detect_feature_datasets(version_filter=self.args.dataset_version)
            if not feature_datasets:
                # 显示可用版本
                available_datasets = detect_feature_datasets()
                if available_datasets:
                    self.logger.info("可用的数据集版本:")
                    for dataset in available_datasets:
                        exclude_info = f", 排除: {dataset['exclude_patterns']}" if dataset['exclude_patterns'] else ""
                        self.logger.info(f"  - {dataset['name']} ({dataset['version']}) - {dataset['sample_count']} 张{exclude_info}")
                
                raise FileNotFoundError(f"❌ 未找到版本 {self.args.dataset_version} 的特征数据集！请先生成该版本数据集")
            
            # 使用检测到的数据集
            dataset_info = feature_datasets[0]
            self.args.feature_dataset_path = dataset_info['path']
            
            self.logger.info(f"✅ 选择数据集: {dataset_info['name']} ({dataset_info['version']})")
            self.logger.info(f"   路径: {dataset_info['path']}")
            self.logger.info(f"   样本数量: {dataset_info['sample_count']}")
            if dataset_info['exclude_patterns']:
                self.logger.info(f"   排除模式: {dataset_info['exclude_patterns']}")
            if dataset_info['yolo_model_used']:
                self.logger.info(f"   生成模型: {os.path.basename(dataset_info['yolo_model_used'])}")
            
            # 保存数据集信息
            self.dataset_info = dataset_info
        else:
            self.logger.info(f"使用指定数据集: {self.args.feature_dataset_path}")
            self.dataset_info = None
        
        # 解析细分训练模式
        bg_filter, power_filter = self._parse_training_mode()
        
        # 创建完整数据集
        self.logger.info("创建数据加载器...")
        full_dataloader, full_dataset = create_feature_dataloader(
            feature_dataset_path=self.args.feature_dataset_path,
            batch_size=self.args.batch_size,
            shuffle=False,  # 先不打乱，方便分割
            bg_type=bg_filter,
            power_filter=power_filter,
            concentration_range=self.args.concentration_range,
            image_size=self.args.image_size
        )
        
        self.logger.info(f"数据集加载完成，总样本数: {len(full_dataset)}")
        
        # 分割训练集和验证集
        self.logger.info("=== 第2阶段：数据集分割 ===")
        
        # 导入数据集划分工具
        import sys
        sys.path.append('../src')  # 添加src目录到路径以导入dataset_split_utils
        from dataset_split_utils import split_dataset
        
        # 根据指定方法划分数据集
        if self.args.split_method == 'random':
            self.logger.info(f"使用随机划分方式 (训练比例: {self.args.train_ratio})")
            train_dataset, val_dataset = split_dataset(
                full_dataset, 
                split_method='random', 
                train_ratio=self.args.train_ratio,
                seed=42
            )
        elif self.args.split_method == 'stratified':
            self.logger.info(f"使用按浓度分层抽样划分方式 (训练比例: {self.args.train_ratio})")
            train_dataset, val_dataset = split_dataset(
                full_dataset, 
                split_method='stratified', 
                train_ratio=self.args.train_ratio,
                seed=42
            )
        elif self.args.split_method == 'interval':
            self.logger.info(f"使用按浓度间隔分配划分方式 (训练间隔长度: {self.args.train_interval_length}, 验证间隔长度: {self.args.val_interval_length})")
            train_dataset, val_dataset = split_dataset(
                full_dataset, 
                split_method='interval', 
                train_interval_length=self.args.train_interval_length,
                val_interval_length=self.args.val_interval_length,
                seed=42
            )
        
        train_size = len(train_dataset)
        val_size = len(val_dataset)
        total_size = train_size + val_size
        
        self.logger.info(f"数据集分割配置:")
        self.logger.info(f"   总样本数: {total_size}")
        self.logger.info(f"   训练集大小: {train_size}")
        self.logger.info(f"   验证集大小: {val_size}")
        
        # 创建DataLoader
        self.logger.info("创建数据加载器...")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=safe_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=safe_collate_fn
        )
        
        self.logger.info(f"DataLoader创建完成:")
        self.logger.info(f"   训练批次数: {len(self.train_loader)}")
        self.logger.info(f"   验证批次数: {len(self.val_loader)}")
        
        # 获取数据集统计信息
        self.logger.info("=== 第3阶段：数据集统计分析 ===")
        conc_stats = full_dataset.get_concentration_statistics()
        meta_stats = full_dataset.get_metadata_statistics()
        
        self.logger.info(f"浓度统计:")
        self.logger.info(f"   范围: {conc_stats['min']:.1f} - {conc_stats['max']:.1f}")
        self.logger.info(f"   平均: {conc_stats['mean']:.2f} ± {conc_stats['std']:.2f}")
        self.logger.info(f"   种类: {len(conc_stats['unique_values'])} 种")
        
        self.logger.info(f"元数据统计:")
        self.logger.info(f"   背景分布: {meta_stats['bg_types']}")
        self.logger.info(f"   功率分布: {meta_stats['powers']}")
        
        self.logger.info("=== 数据加载阶段完成 ===\n")
        return full_dataset
    
    def create_model(self):
        """创建模型"""
        self.logger.info(f"=== 第4阶段：创建{self.args.model_type.upper()}模型 ===")
        
        if self.args.model_type == 'cnn':
            self.logger.info("初始化AdvancedLaserSpotCNN模型...")
            self.model = AdvancedLaserSpotCNN(
                num_features=self.args.hidden_dim,
                use_attention=True,
                use_multiscale=True,
                use_laser_attention=False  # 特征数据集上禁用激光光斑专用注意力
            )
        elif self.args.model_type == 'vgg':
            self.logger.info("初始化VGG+CBAM模型...")
            self.model = VGGRegressionCBAM(
                freeze_features=self.args.freeze_features,  # 改为可配置，默认解冻
                debug_mode=False  # 禁用调试模式
            )
        elif self.args.model_type == 'resnet50':
            self.logger.info("初始化ResNet50回归模型...")
            self.model = ResNet50Regression(
                freeze_backbone=self.args.freeze_backbone,  # 改为可配置，默认解冻
                dropout_rate=0.5  # Dropout比率
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.args.model_type}")
        
        self.model = self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型架构分析:")
        if self.args.model_type == 'cnn':
            self.logger.info(f"   模型类型: 高级CNN (残差+注意力+多尺度)")
            self.logger.info(f"   激光光斑专用注意力: 已禁用（适配特征数据集）")
        elif self.args.model_type == 'vgg':
            self.logger.info(f"   模型类型: VGG16+CBAM (预训练+注意力)")
            freeze_status = "已冻结" if self.args.freeze_features else "已解冻"
            self.logger.info(f"   预训练特征: {freeze_status}（ImageNet权重）")
        elif self.args.model_type == 'resnet50':
            self.logger.info(f"   模型类型: ResNet50回归 (预训练主干+回归头)")
            freeze_status = "已冻结" if self.args.freeze_backbone else "已解冻"
            self.logger.info(f"   主干网络: {freeze_status}（ImageNet权重）")
            self.logger.info(f"   回归头: 三层全连接+Dropout+ReLU")
        self.logger.info(f"   总参数: {total_params:,}")
        self.logger.info(f"   可训练参数: {trainable_params:,}")
        self.logger.info(f"   模型大小: {total_params * 4 / 1024**2:.1f} MB")
        
        # 测试模型前向传播
        self.logger.info("测试模型前向传播...")
        test_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            test_output = self.model(test_input)
        self.logger.info(f"   输入形状: {test_input.shape}")
        # 处理tuple输出的情况
        if isinstance(test_output, tuple):
            self.logger.info(f"   输出形状: {test_output[0].shape} (主输出)")
            if len(test_output) > 1:
                self.logger.info(f"   注意力权重形状: {test_output[1].shape}")
        else:
            self.logger.info(f"   输出形状: {test_output.shape}")
        
        self.logger.info("=== 模型创建完成 ===\n")
    
    def setup_training(self):
        """设置训练组件"""
        self.logger.info("=== 第5阶段：配置训练组件 ===")
        
        # 损失函数 - 可选择是否使用浓度感知损失函数
        use_concentration_aware = not getattr(self.args, 'disable_concentration_aware_loss', False)
        
        if use_concentration_aware:
            threshold = getattr(self.args, 'high_concentration_threshold', 800)
            weight = getattr(self.args, 'high_concentration_weight', 0.5)
            scale = getattr(self.args, 'smooth_scale_factor', 0.01)
            
            # 选择损失函数类型
            if getattr(self.args, 'progressive_weighting', False):
                self.criterion = ProgressiveSmoothWeightedLoss(
                    transition_concentration=threshold,
                    final_beta=weight,
                    warmup_epochs=getattr(self.args, 'warmup_epochs', 30),
                    scale_factor=scale
                )
                self.logger.info(f"损失函数: ProgressiveSmoothWeightedLoss (渐进式平滑权重)")
                self.logger.info(f"   ⏰ 预热轮次: {getattr(self.args, 'warmup_epochs', 30)}")
            elif getattr(self.args, 'use_legacy_loss', False):
                # 使用原有的硬阈值损失函数
                self.criterion = ConcentrationAwareLoss(
                    high_concentration_threshold=threshold,
                    high_concentration_weight=weight
                )
                self.logger.info(f"损失函数: ConcentrationAwareLoss (传统硬阈值版本)")
            else:
                # 默认使用平滑权重损失函数
                self.criterion = SmoothAdaptiveWeightedLoss(
                    transition_concentration=threshold,
                    beta=weight,
                    scale_factor=scale
                )
                self.logger.info(f"损失函数: SmoothAdaptiveWeightedLoss (平滑权重)")
            
            self.logger.info(f"   🎯 高浓度优化: 已启用浓度感知损失函数")
            self.logger.info(f"   📊 过渡浓度: {threshold} mg/L")
            self.logger.info(f"   📊 最小权重: {weight} (降权{(1-weight)*100:.0f}%)")
            self.logger.info(f"   📊 平滑因子: {scale}")
            self.logger.info(f"   📈 预期效果: 高浓度区域损失降权,缓解特征趋同问题")
        else:
            self.criterion = nn.MSELoss()
            self.logger.info(f"损失函数: {type(self.criterion).__name__} (标准MSE)")
            self.logger.info("   ⚠️  浓度感知优化: 已禁用,使用标准MSE损失")
        
        # 优化器
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        
        self.logger.info(f"优化器配置:")
        self.logger.info(f"   类型: {type(self.optimizer).__name__}")
        self.logger.info(f"   学习率: {self.args.learning_rate}")
        self.logger.info(f"   权重衰减: {self.args.weight_decay}")
        
        # 学习率调度器
        if self.args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.step_size,
                gamma=self.args.gamma
            )
            self.logger.info(f"学习率调度器: StepLR (步长={self.args.step_size}, 衰减={self.args.gamma})")
        elif self.args.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.args.gamma,
                patience=self.args.patience,
                min_lr=1e-7
            )
            self.logger.info(f"学习率调度器: ReduceLROnPlateau (耐心={self.args.patience}, 衰减={self.args.gamma})")
        elif self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs
            )
            self.logger.info(f"学习率调度器: CosineAnnealingLR (T_max={self.args.epochs})")
        else:
            self.scheduler = None
            self.logger.info("学习率调度器: 无")
        
        self.logger.info("=== 训练组件配置完成 ===\n")
    
    def _check_disk_space(self, required_mb=500):
        """检查磁盘空间是否充足"""
        import shutil
        try:
            total, used, free = shutil.disk_usage(self.output_dir)
            free_mb = free // (1024 * 1024)
            
            if free_mb < required_mb:
                self.logger.warning(f"⚠️ 磁盘空间不足: 剩余 {free_mb} MB, 需要 {required_mb} MB")
                return False
            
            self.logger.debug(f"磁盘空间充足: 剩余 {free_mb} MB")
            return True
        except Exception as e:
            self.logger.warning(f"无法检查磁盘空间: {e}")
            return True  # 默认允许保存
    
    def _cleanup_old_checkpoints(self, keep_latest=3):
        """清理旧的检查点文件，保留最新的几个"""
        try:
            checkpoint_pattern = os.path.join(self.output_dir, 'checkpoint_epoch_*.pth')
            checkpoints = glob.glob(checkpoint_pattern)
            
            if len(checkpoints) <= keep_latest:
                return
            
            # 按文件修改时间排序，删除最旧的
            checkpoints.sort(key=os.path.getmtime)
            to_delete = checkpoints[:-keep_latest]
            
            deleted_count = 0
            for checkpoint in to_delete:
                try:
                    os.remove(checkpoint)
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"无法删除检查点 {checkpoint}: {e}")
            
            if deleted_count > 0:
                self.logger.info(f"   清理了 {deleted_count} 个旧检查点")
                
        except Exception as e:
            self.logger.warning(f"检查点清理失败: {e}")
    
    def save_checkpoint(self, epoch, train_losses, val_losses, val_maes, val_r2s, best_val_loss, best_epoch):
        """保存训练检查点（含磁盘空间管理）"""
        try:
            # 检查磁盘空间
            if not self._check_disk_space(required_mb=500):
                self.logger.warning("   磁盘空间不足，尝试清理旧检查点...")
                self._cleanup_old_checkpoints(keep_latest=2)
                
                if not self._check_disk_space(required_mb=200):
                    self.logger.error("   磁盘空间仍然不足，跳过检查点保存")
                    return
            
            # 准备检查点数据（减少内存占用）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                # 只保留最近20个损失值，减少文件大小
                'train_losses': [float(x) for x in train_losses[-20:]],
                'val_losses': [float(x) for x in val_losses[-20:]],
                'val_maes': [float(x) for x in val_maes[-20:]],
                'val_r2s': [float(x) for x in val_r2s[-20:]],
                'best_val_loss': float(best_val_loss),
                'best_epoch': int(best_epoch),
                'args': self.args
            }
            
            # 分步保存，避免内存峰值
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            
            # 使用临时文件保证原子性写入
            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint, temp_path)
            
            # 原子性重命名
            os.rename(temp_path, checkpoint_path)
            self.logger.info(f"   检查点已保存: checkpoint_epoch_{epoch+1}.pth")
            
            # 更新最新检查点（软连接或复制）
            latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
            try:
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                
                # 在Windows上使用复制而不是软链接
                import shutil
                shutil.copy2(checkpoint_path, latest_path)
                
            except Exception as e:
                self.logger.warning(f"更新最新检查点失败: {e}")
            
            # 定期清理旧检查点
            if (epoch + 1) % 5 == 0:
                self._cleanup_old_checkpoints(keep_latest=3)
                
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            # 如果是磁盘空间问题，尝试紧急清理
            if "No space left on device" in str(e) or "PytorchStreamWriter failed" in str(e):
                self.logger.warning("检测到磁盘空间问题，执行紧急清理...")
                self._cleanup_old_checkpoints(keep_latest=1)
                
                # 尝试只保存模型权重
                try:
                    model_only_path = os.path.join(self.output_dir, f'model_epoch_{epoch+1}.pth')
                    torch.save(self.model.state_dict(), model_only_path)
                    self.logger.info(f"   仅保存模型权重: model_epoch_{epoch+1}.pth")
                except Exception as e2:
                    self.logger.error(f"紧急保存也失败: {e2}")
            else:
                raise
    
    def load_checkpoint(self, checkpoint_path):
        """加载训练检查点，支持架构兼容性处理"""
        if os.path.isdir(checkpoint_path):
            # 如果是目录，查找最新的检查点
            latest_checkpoint = os.path.join(checkpoint_path, 'checkpoint_latest.pth')
            if os.path.exists(latest_checkpoint):
                checkpoint_path = latest_checkpoint
            else:
                # 查找最新的epoch检查点
                checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                    checkpoint_path = os.path.join(checkpoint_path, checkpoints[-1])
                else:
                    raise FileNotFoundError(f"在目录 {checkpoint_path} 中未找到检查点文件")
        
        self.logger.info(f"正在加载检查点: {checkpoint_path}")
        
        # PyTorch 2.6兼容性修复: 处理weights_only默认值变更
        try:
            # 尝试使用新的安全加载模式
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            if "Weights only load failed" in str(e) or "argparse.Namespace" in str(e):
                # 对于包含argparse.Namespace的检查点，添加安全全局变量
                import argparse
                torch.serialization.add_safe_globals([argparse.Namespace])
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.logger.info("   ✓ 使用安全全局变量模式加载")
            else:
                # 其他错误，重新抛出
                raise e
        
        # 智能加载模型状态，处理架构不匹配
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("   ✓ 完全匹配加载")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                self.logger.warning("   ⚠️ 检测到模型架构不匹配，尝试部分加载...")
                self._load_compatible_state_dict(checkpoint['model_state_dict'])
            else:
                raise e
        
                # 智能加载优化器和调度器状态
        self._smart_load_optimizer_scheduler(checkpoint)
        
        # 兼容处理：检查训练历史字段是否存在
        checkpoint_type = "完整检查点" if 'train_losses' in checkpoint else "最佳模型文件"
        self.logger.info(f"   检查点类型: {checkpoint_type}")
        
        # 为缺少的训练历史字段提供默认值
        for field, default in [
            ('train_losses', []),
            ('val_losses', []),
            ('val_maes', []),
            ('val_r2s', []),
            ('best_val_loss', float('inf')),
            ('best_epoch', 0)
        ]:
            if field not in checkpoint:
                checkpoint[field] = default
                self.logger.warning(f"   ⚠️ 检查点中缺少'{field}'字段，使用默认值: {default}")
        
        self.logger.info(f"检查点加载完成:")
        self.logger.info(f"   恢复到epoch: {checkpoint['epoch'] + 1}")
        self.logger.info(f"   历史最佳验证损失: {checkpoint['best_val_loss']}")
        self.logger.info(f"   历史最佳epoch: {checkpoint['best_epoch'] + 1}")
        
        return checkpoint
    
    def _smart_load_optimizer_scheduler(self, checkpoint):
        """智能加载优化器和调度器状态"""
        self.logger.info("=== 智能参数加载策略 ===")
        
        # 情况1: 显式要求重置优化器
        if self.args.reset_optimizer_on_resume:
            self.logger.info("✨ 显式重置优化器状态（使用新学习率等参数）")
            self._reset_optimizer_to_new_params()
        else:
            # 情况2: 尝试加载检查点中的优化器状态
            try:
                if checkpoint.get('optimizer_state_dict'):
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    restored_lr = self.optimizer.param_groups[0]['lr']
                    
                    self.logger.info(f"✓ 优化器状态已从检查点恢复")
                    self.logger.info(f"   学习率: {old_lr:.2e} -> {restored_lr:.2e}")
                    
                    # 显示已恢复的参数
                    self._log_optimizer_state()
                else:
                    self.logger.warning("⚠️ 检查点中无优化器状态，使用新参数")
                    self._reset_optimizer_to_new_params()
            except Exception as e:
                self.logger.warning(f"⚠️ 优化器状态加载失败，使用新参数: {str(e)}")
                self._reset_optimizer_to_new_params()

        # 情况1: 显式要求重置调度器  
        if self.args.reset_scheduler_on_resume:
            self.logger.info("✨ 显式重置调度器状态（使用新调度器配置）")
            self._reset_scheduler_to_new_params()
        else:
            # 情况2: 尝试加载检查点中的调度器状态
            try:
                if self.scheduler and checkpoint.get('scheduler_state_dict'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    self.logger.info("✓ 调度器状态已从检查点恢复")
                    self._log_scheduler_state()
                else:
                    if self.scheduler:
                        self.logger.warning("⚠️ 检查点中无调度器状态，使用新配置")
                        self._reset_scheduler_to_new_params()
            except Exception as e:
                self.logger.warning(f"⚠️ 调度器状态加载失败，使用新配置: {str(e)}")
                if self.scheduler:
                    self._reset_scheduler_to_new_params()
        
        self.logger.info("=== 参数加载完成 ===")

    def _reset_optimizer_to_new_params(self):
        """重置优化器到新参数"""
        if hasattr(self, '_parameter_changes_detected') and self._parameter_changes_detected:
            self.logger.info(f"   应用新参数: {', '.join(self._parameter_changes_detected)}")
        
        # 重新设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.args.learning_rate
            if hasattr(self.args, 'weight_decay'):
                param_group['weight_decay'] = self.args.weight_decay
        
        self.logger.info(f"   学习率设置为: {self.args.learning_rate:.2e}")
        self.logger.info(f"   权重衰减设置为: {self.args.weight_decay:.2e}")

    def _reset_scheduler_to_new_params(self):
        """重置调度器到新参数"""
        # 重新创建调度器以使用新参数
        if self.args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.step_size,
                gamma=self.args.gamma
            )
            self.logger.info(f"   StepLR重置: step_size={self.args.step_size}, gamma={self.args.gamma}")
        elif self.args.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.args.gamma,
                patience=self.args.patience,
                min_lr=1e-7
            )
            self.logger.info(f"   ReduceLROnPlateau重置: patience={self.args.patience}, factor={self.args.gamma}")
        elif self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs
            )
            self.logger.info(f"   CosineAnnealingLR重置: T_max={self.args.epochs}")
    
    def _log_optimizer_state(self):
        """记录优化器状态"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.logger.info(f"   参数组{i}: lr={param_group['lr']:.2e}, weight_decay={param_group.get('weight_decay', 0):.2e}")
    
    def _log_scheduler_state(self):
        """记录调度器状态"""
        if self.scheduler:
            scheduler_type = type(self.scheduler).__name__
            self.logger.info(f"   调度器类型: {scheduler_type}")
            if hasattr(self.scheduler, 'last_epoch'):
                self.logger.info(f"   当前epoch: {self.scheduler.last_epoch}")
            if hasattr(self.scheduler, 'patience'):
                self.logger.info(f"   patience: {self.scheduler.patience}")
    
    def _initialize_progressive_unfreeze(self):
        """初始化渐进式解冻策略"""
        self.logger.info("=== 初始化渐进式解冻策略 ===")
        
        # 检测数据集类型，针对bg1_400mw调整策略
        is_bg1_400mw = hasattr(self.args, 'bg_mode') and 'bg1_400mw' in str(self.args.bg_mode)
        
        # 定义解冻阶段
        total_epochs = self.args.epochs
        
        if is_bg1_400mw:
            # bg1_400mw专用策略：跳过第一阶段，直接从Layer4解冻开始
            self.unfreeze_stages = {
                'stage1_end': 0,                                 # 跳过第一阶段
                'stage2_end': int(total_epochs * 0.4),           # 前40%轮次  
                'stage3_end': int(total_epochs * 0.7),           # 前70%轮次
                'stage4_end': total_epochs                       # 全部轮次
            }
            self.logger.info("🎯 检测到bg1_400mw数据集，使用专用渐进式解冻策略（跳过回归头阶段）")
        else:
            # 标准策略
            self.unfreeze_stages = {
                'stage1_end': int(total_epochs * 0.2),  # 前20%轮次
                'stage2_end': int(total_epochs * 0.4),  # 前40%轮次  
                'stage3_end': int(total_epochs * 0.6),  # 前60%轮次
                'stage4_end': total_epochs              # 全部轮次
            }
            self.logger.info("📋 使用标准渐进式解冻策略")
        
        self.logger.info(f"渐进式解冻计划:")
        if is_bg1_400mw:
            self.logger.info(f"   阶段1: 跳过（避免回归头不收敛）")
            self.logger.info(f"   阶段2 (轮次1-{self.unfreeze_stages['stage2_end']}): 解冻Layer4")
            self.logger.info(f"   阶段3 (轮次{self.unfreeze_stages['stage2_end']+1}-{self.unfreeze_stages['stage3_end']}): 解冻Layer3-4")
            self.logger.info(f"   阶段4 (轮次{self.unfreeze_stages['stage3_end']+1}-{self.unfreeze_stages['stage4_end']}): 全部解冻")
        else:
            self.logger.info(f"   阶段1 (轮次1-{self.unfreeze_stages['stage1_end']}): 只训练回归头")
            self.logger.info(f"   阶段2 (轮次{self.unfreeze_stages['stage1_end']+1}-{self.unfreeze_stages['stage2_end']}): 解冻Layer4")
            self.logger.info(f"   阶段3 (轮次{self.unfreeze_stages['stage2_end']+1}-{self.unfreeze_stages['stage3_end']}): 解冻Layer3-4")
            self.logger.info(f"   阶段4 (轮次{self.unfreeze_stages['stage3_end']+1}-{self.unfreeze_stages['stage4_end']}): 全部解冻")
        
        # 记录初始学习率和数据集类型
        self.initial_lr = self.optimizer.param_groups[0]['lr']
        self.current_stage = 0
        self.is_bg1_400mw = is_bg1_400mw
        
        # 根据策略设置初始状态
        if is_bg1_400mw:
            # bg1_400mw直接从阶段2开始，解冻Layer4
            self._unfreeze_layer4()
            self.current_stage = 2  # 直接设为阶段2
            self.logger.info("🚀 bg1_400mw策略：直接从阶段2开始（Layer4解冻）")
        else:
            # 标准策略：确保开始时只有回归头可训练
            self._freeze_all_backbone()
            
        self.logger.info("=== 渐进式解冻初始化完成 ===")
    
    def _freeze_all_backbone(self):
        """冻结所有backbone层"""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        self._log_trainable_params("冻结所有backbone后")
    
    def _apply_progressive_unfreeze(self, epoch):
        """应用渐进式解冻策略"""
        new_stage = self._get_current_stage(epoch)
        
        if new_stage != self.current_stage:
            self.logger.info(f"=== 渐进式解冻：进入阶段{new_stage} ===")
            
            if new_stage == 1:
                # 阶段1：只训练回归头（已经设置）
                self._freeze_all_backbone()
                self._adjust_learning_rate(1.0)
                self.logger.info("   策略：只训练回归头")
                
            elif new_stage == 2:
                # 阶段2：解冻Layer4
                self._unfreeze_layer4()
                self._adjust_learning_rate(0.5)
                self.logger.info("   策略：解冻Layer4，学习率减半")
                
            elif new_stage == 3:
                # 阶段3：解冻Layer3-4
                self._unfreeze_layer3_4()
                self._adjust_learning_rate(0.3)
                self.logger.info("   策略：解冻Layer3-4，学习率降至30%")
                
            elif new_stage == 4:
                # 阶段4：全部解冻
                self._unfreeze_all_layers()
                self._adjust_learning_rate(0.1)
                self.logger.info("   策略：全部解冻，学习率降至10%")
            
            self.current_stage = new_stage
            self.logger.info("=== 渐进式解冻阶段切换完成 ===")
    
    def _get_current_stage(self, epoch):
        """获取当前应该处于的阶段"""
        # bg1_400mw跳过阶段1
        if hasattr(self, 'is_bg1_400mw') and self.is_bg1_400mw:
            if epoch < self.unfreeze_stages['stage2_end']:
                return 2  # 直接从阶段2开始
            elif epoch < self.unfreeze_stages['stage3_end']:
                return 3
            else:
                return 4
        else:
            # 标准策略
            if epoch < self.unfreeze_stages['stage1_end']:
                return 1
            elif epoch < self.unfreeze_stages['stage2_end']:
                return 2
            elif epoch < self.unfreeze_stages['stage3_end']:
                return 3
            else:
                return 4
    
    def _unfreeze_layer4(self):
        """解冻ResNet50的Layer4"""
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layer4'):
            for param in self.model.backbone.layer4.parameters():
                param.requires_grad = True
        self._log_trainable_params("解冻Layer4后")
    
    def _unfreeze_layer3_4(self):
        """解冻ResNet50的Layer3和Layer4"""
        if hasattr(self.model, 'backbone'):
            if hasattr(self.model.backbone, 'layer3'):
                for param in self.model.backbone.layer3.parameters():
                    param.requires_grad = True
            if hasattr(self.model.backbone, 'layer4'):
                for param in self.model.backbone.layer4.parameters():
                    param.requires_grad = True
        self._log_trainable_params("解冻Layer3-4后")
    
    def _unfreeze_all_layers(self):
        """解冻所有层"""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
        self._log_trainable_params("全部解冻后")
    
    def _adjust_learning_rate(self, factor):
        """调整学习率"""
        # 针对bg1_400mw使用更温和的学习率调整
        if hasattr(self, 'is_bg1_400mw') and self.is_bg1_400mw:
            # bg1_400mw使用更保守的学习率衰减
            adjusted_factor = max(factor, 0.3)  # 最低不低于30%
            new_lr = self.initial_lr * adjusted_factor
        else:
            new_lr = self.initial_lr * factor
            
        old_lr = self.optimizer.param_groups[0]['lr']
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        dataset_info = " (bg1_400mw优化)" if hasattr(self, 'is_bg1_400mw') and self.is_bg1_400mw else ""
        self.logger.info(f"   学习率调整: {old_lr:.6f} -> {new_lr:.6f} (因子: {factor}){dataset_info}")
    
    def _log_trainable_params(self, context=""):
        """记录可训练参数统计"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.logger.info(f"   {context}参数统计:")
        self.logger.info(f"     总参数: {total_params:,}")
        self.logger.info(f"     可训练: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        self.logger.info(f"     冻结: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    def _load_compatible_state_dict(self, checkpoint_state_dict):
        """兼容性加载模型状态字典"""
        model_state_dict = self.model.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for key, value in checkpoint_state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    model_state_dict[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (形状不匹配: {value.shape} -> {model_state_dict[key].shape})")
            else:
                skipped_keys.append(f"{key} (当前模型中不存在)")
        
        # 加载兼容的参数
        self.model.load_state_dict(model_state_dict)
        
        self.logger.info(f"   兼容性加载统计:")
        self.logger.info(f"     成功加载: {len(loaded_keys)} 个参数")
        self.logger.info(f"     跳过参数: {len(skipped_keys)} 个")
        
        if skipped_keys:
            self.logger.warning("   跳过的参数:")
            for key in skipped_keys[:5]:  # 只显示前5个
                self.logger.warning(f"     - {key}")
            if len(skipped_keys) > 5:
                self.logger.warning(f"     ... 及其他 {len(skipped_keys) - 5} 个参数")
        
        # 计算加载比例
        load_ratio = len(loaded_keys) / len(checkpoint_state_dict) * 100
        self.logger.info(f"   参数加载比例: {load_ratio:.1f}%")
        
        if load_ratio < 50:
            self.logger.error("   ❌ 加载比例过低，可能影响训练效果")
            raise RuntimeError("检查点兼容性太差，建议从头开始训练")
        elif load_ratio < 80:
            self.logger.warning("   ⚠️ 加载比例较低，建议降低学习率")
        else:
            self.logger.info("   ✓ 加载比例良好，可以继续训练")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        # 更新渐进式权重损失函数的epoch
        if hasattr(self.criterion, 'update_epoch'):
            self.criterion.update_epoch(epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        # 进度监控
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # 每10%输出一次
        
        for batch_idx, (images, targets, metadata) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            # 处理模型输出（可能是tuple）
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 取主输出
            
            # 针对不同模型调整目标值形状以避免广播警告
            if self.args.model_type == 'resnet50':
                # ResNet50输出标量，目标值也应为标量
                loss = self.criterion(outputs.squeeze(), targets.squeeze())
            else:
                # 其他模型保持原有处理方式
                loss = self.criterion(outputs.squeeze(), targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 详细进度输出
            if batch_idx % log_interval == 0:
                progress = (batch_idx + 1) / total_batches * 100
                avg_loss = total_loss / num_batches
                self.logger.info(f"   训练进度: {batch_idx+1}/{total_batches} ({progress:.1f}%) - "
                               f"当前损失: {loss.item():.6f}, 平均损失: {avg_loss:.6f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """验证模型"""
        self.logger.info("   开始验证...")
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        val_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, (images, batch_targets, metadata) in enumerate(self.val_loader):
                images = images.to(self.device)
                batch_targets = batch_targets.to(self.device).float()
                
                outputs = self.model(images)
                # 处理模型输出（可能是tuple）
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取主输出
                
                # 针对不同模型调整目标值形状以避免广播警告
                if self.args.model_type == 'resnet50':
                    # ResNet50输出标量，目标值也应为标量
                    loss = self.criterion(outputs.squeeze(), batch_targets.squeeze())
                else:
                    # 其他模型保持原有处理方式
                    loss = self.criterion(outputs.squeeze(), batch_targets)
                
                total_loss += loss.item()
                
                # 安全处理预测值和目标值，避免0维数组问题
                pred_numpy = outputs.squeeze().cpu().numpy()
                target_numpy = batch_targets.cpu().numpy()
                
                # 确保是1维数组，即使batch_size=1
                if pred_numpy.ndim == 0:
                    pred_numpy = np.array([pred_numpy])
                if target_numpy.ndim == 0:
                    target_numpy = np.array([target_numpy])
                    
                predictions.extend(pred_numpy)
                targets.extend(target_numpy)
                
                # 验证进度
                if batch_idx % max(1, val_batches // 5) == 0:
                    progress = (batch_idx + 1) / val_batches * 100
                    self.logger.info(f"   验证进度: {batch_idx+1}/{val_batches} ({progress:.1f}%)")
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算评估指标
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        self.logger.info(f"   验证完成: 损失={avg_loss:.6f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return avg_loss, mse, mae, r2, predictions, targets
    
    def train(self):
        """训练模型"""
        self.logger.info("=== 第6阶段：开始模型训练 ===")
        self.logger.info(f"训练配置:")
        self.logger.info(f"   总轮次: {self.args.epochs}")
        self.logger.info(f"   批次大小: {self.args.batch_size}")
        self.logger.info(f"   训练批次: {len(self.train_loader)}")
        self.logger.info(f"   验证批次: {len(self.val_loader)}")
        
        # 初始化渐进式解冻
        if self.args.progressive_unfreeze and self.args.model_type == 'resnet50':
            self._initialize_progressive_unfreeze()
        
        # 训练历史
        train_losses = []
        val_losses = []
        val_maes = []
        val_r2s = []
        
        best_val_loss = float('inf')
        best_epoch = 0
        start_epoch = 0
        
        # 检查是否需要从检查点恢复
        if self.args.resume:
            checkpoint = self.load_checkpoint(self.args.resume)
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            val_maes = checkpoint['val_maes']
            val_r2s = checkpoint['val_r2s']
            best_val_loss = checkpoint['best_val_loss']
            best_epoch = checkpoint['best_epoch']
            self.logger.info(f"从第 {start_epoch + 1} 轮次继续训练")
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, self.args.epochs):
            epoch_start_time = time.time()
            self.logger.info(f"\n--- 轮次 {epoch+1}/{self.args.epochs} ---")
            
            # 执行渐进式解冻策略
            if self.args.progressive_unfreeze and self.args.model_type == 'resnet50':
                self._apply_progressive_unfreeze(epoch)
            
            # 训练
            self.logger.info("训练阶段:")
            train_loss = self.train_epoch(epoch)
            
            # 验证
            self.logger.info("验证阶段:")
            val_loss, val_mse, val_mae, val_r2, predictions, targets = self.validate()
            
            # 更新学习率
            if self.scheduler:
                old_lr = self.optimizer.param_groups[0]['lr']
                
                # 根据调度器类型使用不同的step方法
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # ReduceLROnPlateau需要传入损失值
                else:
                    self.scheduler.step()  # 其他调度器不需要参数
                    
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    self.logger.info(f"   学习率更新: {old_lr:.6f} -> {new_lr:.6f}")
                
                # 检查学习率是否过小
                if new_lr < 1e-7:
                    self.logger.warning(f"   ⚠️ 学习率过小 ({new_lr:.2e})，训练可能停滞")
            
            # 记录历史
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_maes.append(val_mae)
            val_r2s.append(val_r2)
            
            # 轮次总结
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(f"轮次 {epoch+1} 总结:")
            self.logger.info(f"   训练损失: {train_loss:.6f}")
            self.logger.info(f"   验证损失: {val_loss:.6f}")
            self.logger.info(f"   验证MAE: {val_mae:.4f}")
            self.logger.info(f"   验证R²: {val_r2:.4f}")
            self.logger.info(f"   当前学习率: {current_lr:.6f}")
            self.logger.info(f"   轮次耗时: {epoch_time:.1f}s")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                
                # 保存完整的检查点格式，确保兼容性
                best_model_checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'args': self.args
                }
                
                torch.save(best_model_checkpoint, 
                          os.path.join(self.output_dir, 'best_model.pth'))
                self.logger.info(f"   ✓ 新的最佳模型 (验证损失: {val_loss:.6f})")
            else:
                improvement = val_loss - best_val_loss
                self.logger.info(f"   当前模型比最佳模型差 {improvement:.6f}")
            
            # 保存检查点
            if (epoch + 1) % self.args.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, train_losses, val_losses, val_maes, val_r2s, best_val_loss, best_epoch)
        
        total_time = time.time() - training_start_time
        self.logger.info(f"\n=== 训练完成 ===")
        self.logger.info(f"训练总结:")
        self.logger.info(f"   最佳轮次: {best_epoch+1}")
        self.logger.info(f"   最佳验证损失: {best_val_loss:.6f}")
        self.logger.info(f"   最终验证R²: {val_r2s[-1]:.4f}")
        self.logger.info(f"   总训练时间: {total_time:.1f}s ({total_time/60:.1f}分钟)")
        self.logger.info(f"   平均每轮时间: {total_time/self.args.epochs:.1f}s")
        
        # 保存训练历史
        history = {
            'train_losses': [float(x) for x in train_losses],  # 转换numpy float32为Python float
            'val_losses': [float(x) for x in val_losses],
            'val_maes': [float(x) for x in val_maes],
            'val_r2s': [float(x) for x in val_r2s],
            'best_epoch': int(best_epoch),
            'best_val_loss': float(best_val_loss),
            'total_training_time': float(total_time)
        }
        
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves(history)
        
        return history
    
    def plot_training_curves(self, history):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='训练损失')
        axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='验证损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE曲线
        axes[0, 1].plot(epochs, history['val_maes'], 'g-', label='验证MAE')
        axes[0, 1].set_title('验证平均绝对误差')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # R²曲线
        axes[1, 0].plot(epochs, history['val_r2s'], 'm-', label='验证R²')
        axes[1, 0].set_title('验证决定系数')
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 最后一次验证的预测vs真实值
        # 重新加载最佳模型进行最终评估
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            try:
                # PyTorch 2.6兼容性修复: 处理weights_only默认值变更
                try:
                    # 尝试使用新的安全加载模式
                    best_checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                except Exception as e:
                    if "Weights only load failed" in str(e) or "argparse.Namespace" in str(e):
                        # 对于包含argparse.Namespace的检查点，添加安全全局变量
                        import argparse
                        torch.serialization.add_safe_globals([argparse.Namespace])
                        best_checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
                        self.logger.info("   ✓ 使用安全全局变量模式加载最佳模型")
                    else:
                        # 其他错误，重新抛出
                        raise e
                
                # 检查是否是新格式（包含'model_state_dict'键）
                if isinstance(best_checkpoint, dict) and 'model_state_dict' in best_checkpoint:
                    self.model.load_state_dict(best_checkpoint['model_state_dict'])
                else:
                    # 旧格式，直接是state_dict
                    self.model.load_state_dict(best_checkpoint)
                _, _, _, _, final_predictions, final_targets = self.validate()
            except Exception as e:
                self.logger.warning(f"加载最佳模型失败，使用当前模型状态: {e}")
                _, _, _, _, final_predictions, final_targets = self.validate()
        else:
            self.logger.warning("未找到最佳模型文件，使用当前模型状态")
            _, _, _, _, final_predictions, final_targets = self.validate()
        
        axes[1, 1].scatter(final_targets, final_predictions, alpha=0.6)
        axes[1, 1].plot([min(final_targets), max(final_targets)], 
                        [min(final_targets), max(final_targets)], 'r--', label='理想预测')
        axes[1, 1].set_title('预测值 vs 真实值')
        axes[1, 1].set_xlabel('真实浓度')
        axes[1, 1].set_ylabel('预测浓度')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300)
        plt.close()
        
        self.logger.info("训练曲线已保存: training_curves.png")
    
    def save_training_info(self, dataset, history):
        """保存训练信息"""
        info = {
            'training_time': datetime.now().isoformat(),
            'feature_dataset_path': self.args.feature_dataset_path,
            'model_type': self.args.model_type,
            'training_args': vars(self.args),
            'dataset_info': {
                'total_samples': len(dataset),
                'concentration_stats': dataset.get_concentration_statistics(),
                'metadata_stats': dataset.get_metadata_statistics()
            },
            'training_results': {
                'best_epoch': history['best_epoch'],
                'best_val_loss': history['best_val_loss'],
                'final_val_mae': history['val_maes'][-1],
                'final_val_r2': history['val_r2s'][-1]
            }
        }
        
        info_path = os.path.join(self.output_dir, 'training_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        self.logger.info("训练信息已保存: training_info.json")

    def train_epoch_with_dual_eval(self, epoch):
        """训练一个epoch，同时跟踪平滑权重损失和标准MSE"""
        self.model.train()
        
        # 更新渐进式权重
        if hasattr(self.criterion, 'update_epoch'):
            self.criterion.update_epoch(epoch)
        
        total_smooth_loss = 0.0
        total_standard_loss = 0.0
        num_batches = 0
        
        # 标准MSE用于性能评估
        standard_mse = nn.MSELoss()
        
        # 进度监控
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # 每10%输出一次
        
        for batch_idx, (images, targets, metadata) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            # 处理模型输出（可能是tuple）
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 取主输出
            
            # 用于训练的平滑权重损失
            smooth_loss = self.criterion(outputs.squeeze(), targets.squeeze())
            
            # 用于评估的标准MSE（不参与梯度更新）
            with torch.no_grad():
                standard_loss = standard_mse(outputs.squeeze(), targets.squeeze())
            
            smooth_loss.backward()
            self.optimizer.step()
            
            total_smooth_loss += smooth_loss.item()
            total_standard_loss += standard_loss.item()
            num_batches += 1
            
            # 进度输出
            if batch_idx % log_interval == 0:
                progress = 100.0 * batch_idx / total_batches
                self.logger.info(f"   批次 {batch_idx:3d}/{total_batches} ({progress:5.1f}%) | "
                               f"平滑损失: {smooth_loss.item():.4f} | "
                               f"标准MSE: {standard_loss.item():.4f}")
        
        avg_smooth_loss = total_smooth_loss / num_batches
        avg_standard_loss = total_standard_loss / num_batches
        
        return avg_smooth_loss, avg_standard_loss

    def validate_with_dual_eval(self):
        """验证模型，返回平滑权重损失和标准MSE"""
        self.model.eval()
        total_smooth_loss = 0.0
        total_standard_loss = 0.0
        predictions = []
        targets = []
        
        standard_mse = nn.MSELoss()
        
        with torch.no_grad():
            for images, batch_targets, metadata in self.val_loader:
                images = images.to(self.device)
                batch_targets = batch_targets.to(self.device).float()
                
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # 平滑权重损失
                smooth_loss = self.criterion(outputs.squeeze(), batch_targets.squeeze())
                
                # 标准MSE
                standard_loss = standard_mse(outputs.squeeze(), batch_targets.squeeze())
                
                total_smooth_loss += smooth_loss.item()
                total_standard_loss += standard_loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        avg_smooth_loss = total_smooth_loss / len(self.val_loader)
        avg_standard_loss = total_standard_loss / len(self.val_loader)
        
        # 计算其他指标
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return (avg_smooth_loss, avg_standard_loss, mae, r2, predictions, targets)


def main():
    parser = argparse.ArgumentParser(description='使用特征数据集训练回归模型')
    
    # 数据集参数
    parser.add_argument('--feature_dataset_path', type=str, default=None,
                       help='特征数据集路径 (默认自动检测)')
    parser.add_argument('--dataset_version', type=str, default='latest',
                       choices=['v1', 'v2', 'v3', 'v4', 'latest'],
                       help='特征数据集版本 (默认latest)')
    parser.add_argument('--bg_mode', type=str, default='all', 
                       choices=['bg0', 'bg1', 'all', 'bg0_20mw', 'bg0_100mw', 'bg0_400mw', 
                               'bg1_20mw', 'bg1_100mw', 'bg1_400mw'],
                       help='训练模式: bg0/bg1/all(传统3档) 或 bg0_20mw等(新6档细分)')
    parser.add_argument('--power_filter', type=str, default=None,
                       help='过滤特定功率 (例如: 20mw, 100mw, 400mw) - 与bg_mode细分冲突时忽略')
    parser.add_argument('--concentration_range', type=str, default=None,
                       help='浓度范围过滤 (格式: "min,max" 例如: "0,1000", 默认: None)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例 (默认: 0.8)')
    parser.add_argument('--split_method', type=str, default='random',
                       choices=['random', 'stratified', 'interval'],
                       help='数据集划分方式: random(随机), stratified(按浓度分层), interval(按浓度间隔)')
    parser.add_argument('--train_interval_length', type=int, default=4,
                       help='训练集浓度间隔长度 (仅在split_method=interval时有效)')
    parser.add_argument('--val_interval_length', type=int, default=1,
                       help='验证集浓度间隔长度 (仅在split_method=interval时有效)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='输入图像尺寸 (默认224)')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'vgg', 'resnet50'],
                       help='模型类型: cnn(高级CNN), vgg(VGG+CBAM), resnet50(ResNet50回归) (默认: cnn)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度 (默认: 512)')
    
    # 冻结权重参数 - 新增
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                       help='是否冻结预训练主干网络 (默认: False, 即解冻)')
    parser.add_argument('--freeze_features', action='store_true', default=False,
                       help='是否冻结VGG预训练特征 (默认: False, 即解冻)')
    parser.add_argument('--progressive_unfreeze', action='store_true', default=False,
                       help='是否使用渐进式解冻策略')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮次 (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='权重衰减 (默认: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='优化器 (默认: adam)')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'plateau', 'cosine', 'none'],
                       help='学习率调度器 (默认: step)')
    parser.add_argument('--step_size', type=int, default=50,
                       help='StepLR步长 (默认: 50)')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='StepLR衰减率 (默认: 0.5)')
    parser.add_argument('--patience', type=int, default=10,
                       help='ReduceLROnPlateau耐心值 (默认: 10)')
    
    # 断点续训参数
    parser.add_argument('--resume', type=str, default=None,
                       help='从指定检查点恢复训练（输出目录路径）')
    parser.add_argument('--reset_optimizer_on_resume', action='store_true', default=False,
                       help='加载检查点时重置优化器状态，使用新的学习率等参数')
    parser.add_argument('--reset_scheduler_on_resume', action='store_true', default=False,
                       help='加载检查点时重置调度器状态')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='早停耐心值 (默认: 15)')
    parser.add_argument('--save_checkpoint_every', type=int, default=10,
                       help='每N个epoch保存一次检查点 (默认: 10)')
    
    # 高浓度优化参数
    parser.add_argument('--disable_concentration_aware_loss', action='store_true',
                       help='禁用浓度感知损失,使用标准MSE')
    parser.add_argument('--high_concentration_threshold', type=float, default=800,
                       help='高浓度阈值 (mg/L), 默认800')
    parser.add_argument('--high_concentration_weight', type=float, default=0.5,
                       help='高浓度损失权重, 默认0.5 (50%% 降权)')
    
    # 平滑权重损失函数参数
    parser.add_argument('--smooth_scale_factor', type=float, default=0.01,
                       help='平滑权重过渡斜率 (默认: 0.01)')
    parser.add_argument('--progressive_weighting', action='store_true',
                       help='启用渐进式权重调整')
    parser.add_argument('--warmup_epochs', type=int, default=30,
                       help='渐进式权重预热轮次 (默认: 30)')
    parser.add_argument('--use_legacy_loss', action='store_true',
                       help='使用传统硬阈值损失函数 (ConcentrationAwareLoss)')
    
    args = parser.parse_args()
    
    print("使用特征数据集训练回归模型")
    print("=" * 50)
    print(f"训练模式: {args.bg_mode.upper()}")
    
    # 输出浓度范围过滤信息
    if args.concentration_range:
        print(f"浓度范围过滤: {args.concentration_range}")
    else:
        print("浓度范围过滤: 未启用")
    
    try:
        # 创建训练器
        trainer = FeatureDatasetTrainer(args)
        
        # 智能检测参数变化（必须在加载数据前执行）
        trainer._detect_parameter_changes()
        
        # 加载数据
        dataset = trainer.load_data()
        
        # 创建模型
        trainer.create_model()
        
        # 设置训练
        trainer.setup_training()
        
        # 训练模型
        history = trainer.train()
        
        # 保存训练信息
        trainer.save_training_info(dataset, history)
        
        print(f"\n训练完成！")
        print(f"结果保存在: {trainer.output_dir}")
        print(f"最佳模型: {trainer.output_dir}/best_model.pth")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()