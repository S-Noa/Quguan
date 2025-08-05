#!/usr/bin/env python3
"""
传统CNN在特征数据集上的训练脚本
基于train_fast.py，适配特征数据集，移除光斑检测功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import argparse
import time
import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import json
from datetime import datetime

# 添加src路径
import sys
sys.path.append('src')

from cnn_model import CNNFeatureExtractor
from feature_dataset_loader import create_feature_dataloader, detect_feature_datasets
from font_utils import get_labels, CHINESE_SUPPORTED, suppress_font_warnings

# 添加数据集名称工具
sys.path.append('.')
from dataset_name_utils import generate_training_output_dir, parse_training_mode_from_args, get_dataset_info_string

# 抑制字体警告
suppress_font_warnings()

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
        
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            self.current_beta = 1.0 - progress * (1.0 - self.final_beta)
        else:
            self.current_beta = self.final_beta
        self.beta = self.current_beta

# 保留原有的ConcentrationAwareLoss作为兼容选项
class ConcentrationAwareLoss(nn.Module):
    """
    浓度感知损失函数 - 方案一:最小改动版本
    解决高浓度特征趋同导致的损失激增问题
    """
    
    def __init__(self, high_concentration_threshold=800, high_concentration_weight=0.5):
        """
        Args:
            high_concentration_threshold: 高浓度阈值(mg/L)，默认800
            high_concentration_weight: 高浓度区域权重，默认0.5(降权50%)
        """
        super(ConcentrationAwareLoss, self).__init__()
        self.threshold = high_concentration_threshold
        self.high_weight = high_concentration_weight
        
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
        
        # 可选:记录损失分解(用于调试)
        if hasattr(self, '_log_loss_components') and self._log_loss_components:
            print(f"损失分解 - 低浓度: {loss_components['low_concentration']:.4f}, "
                  f"高浓度: {loss_components['high_concentration']:.4f}, "
                  f"总计: {total_loss.item():.4f}")
        
        return total_loss

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
            print(f"第一个项目类型: {type(batch[0])}")
            print(f"第一个项目长度: {len(batch[0]) if hasattr(batch[0], '__len__') else 'N/A'}")
        raise

def setup_logger(log_file=None, group_tag=None):
    """设置日志记录"""
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f'baseline_cnn_training_{group_tag}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('Baseline_CNN_Training')
    return logger, log_file

def create_prediction_scatter_plot(targets, predictions, save_path, title="预测vs真实值"):
    """创建预测vs真实值散点图"""
    labels = get_labels(CHINESE_SUPPORTED)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
    
    plt.xlabel(labels['true_concentration'])
    plt.ylabel(labels['predicted_concentration'])
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 计算R²
    r2 = r2_score(targets, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

class BaselineCNNTrainer:
    """传统CNN训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用数据集名称工具生成输出目录
        # 需要先获取数据集路径
        dataset_path = args.feature_dataset_path
        if dataset_path is None:
            # 如果没有指定，先检测可用数据集
            feature_datasets = detect_feature_datasets()
            if feature_datasets:
                dataset_path = feature_datasets[-1]
        
        # 生成基于数据集的输出目录名
        if dataset_path:
            try:
                bg_filter, power_filter, training_mode = parse_training_mode_from_args(args)
                # 添加增量训练标识
                model_name = "baseline_cnn"
                if args.incremental_training:
                    model_name += "_incremental"
                
                self.output_dir = generate_training_output_dir(
                    model_name=model_name,
                    dataset_path=dataset_path,
                    training_mode=training_mode,
                    bg_filter=bg_filter,
                    power_filter=power_filter
                )
            except Exception as e:
                # 如果生成失败，使用传统方式
                print(f"Warning: 无法生成基于数据集的目录名，使用传统方式: {e}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mode_suffix = args.bg_mode.replace('_', '-')
                incremental_suffix = "_incremental" if args.incremental_training else ""
                self.output_dir = f"baseline_cnn{incremental_suffix}_{mode_suffix}_results_{timestamp}"
        else:
            # 使用传统方式
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_suffix = args.bg_mode.replace('_', '-')
            incremental_suffix = "_incremental" if args.incremental_training else ""
            self.output_dir = f"baseline_cnn{incremental_suffix}_{mode_suffix}_results_{timestamp}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        mode_suffix = args.bg_mode.replace('_', '-')
        self.logger, _ = setup_logger(
            os.path.join(self.output_dir, 'training.log'), 
            f'baseline_{mode_suffix}'
        )
        
        self.logger.info(f"传统CNN训练器初始化")
        self.logger.info(f"   设备: {self.device}")
        self.logger.info(f"   训练模式: {args.bg_mode.upper()}")
        self.logger.info(f"   输出目录: {self.output_dir}")
        
        # 增量训练日志
        if args.incremental_training:
            self.logger.info(f"   🔄 增量训练模式: 启用")
            self.logger.info(f"   预训练模型: {args.pretrained_model if args.pretrained_model else '无'}")
            self.logger.info(f"   学习率因子: {args.incremental_lr_factor}")
            self.logger.info(f"   冻结特征层: {'是' if args.freeze_features else '否'}")
        
        # 添加数据集信息日志
        if dataset_path:
            try:
                bg_filter, power_filter, _ = parse_training_mode_from_args(args)
                dataset_info = get_dataset_info_string(dataset_path, bg_filter, power_filter)
                self.logger.info(f"   {dataset_info}")
            except:
                pass
    
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
        
        # 自动检测特征数据集
        if self.args.feature_dataset_path is None:
            self.logger.info("自动检测特征数据集...")
            feature_datasets = detect_feature_datasets()
            if not feature_datasets:
                raise FileNotFoundError("未找到特征数据集！")
            
            self.args.feature_dataset_path = feature_datasets[-1]
            self.logger.info(f"   选择最新数据集: {self.args.feature_dataset_path}")
        
        # 解析细分训练模式
        bg_filter, power_filter = self._parse_training_mode()
        
        # 解析浓度范围参数
        concentration_range = None
        if self.args.concentration_range:
            try:
                min_val, max_val = map(float, self.args.concentration_range.split(','))
                concentration_range = (min_val, max_val)
                self.logger.info(f"应用浓度范围过滤: {min_val:.1f} - {max_val:.1f} mg/L")
            except Exception as e:
                self.logger.warning(f"浓度范围参数解析失败: {e}")
        
        # 创建完整数据集
        self.logger.info("创建数据加载器...")
        full_dataloader, full_dataset = create_feature_dataloader(
            feature_dataset_path=self.args.feature_dataset_path,
            batch_size=self.args.batch_size,
            shuffle=False,  # 先不打乱，方便分割
            bg_type=bg_filter,
            power_filter=power_filter,
            concentration_range=concentration_range
        )
        
        self.logger.info(f"数据集加载完成，总样本数: {len(full_dataset)}")
        
        # 分割训练集和验证集
        self.logger.info("=== 第2阶段：数据集分割 ===")
        
        # 导入数据集划分工具
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
        # Windows系统使用num_workers=0避免多进程问题
        num_workers = 0 if os.name == 'nt' else 2
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=safe_collate_fn  # 使用安全的collate函数
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=safe_collate_fn  # 使用安全的collate函数
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
        """创建CNN模型"""
        self.logger.info("=== 第2阶段：创建CNN模型 ===")
        
        # 创建模型
        self.model = CNNFeatureExtractor().to(self.device)
        
        # 增量训练：加载预训练权重
        if self.args.incremental_training and self.args.pretrained_model:
            self._load_pretrained_weights()
        
        # 冻结特征层（如果启用）
        if self.args.incremental_training and self.args.freeze_features:
            self._freeze_feature_layers()
        
        # 计算模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型创建完成:")
        self.logger.info(f"   总参数数: {total_params:,}")
        self.logger.info(f"   可训练参数: {trainable_params:,}")
        if total_params != trainable_params:
            frozen_params = total_params - trainable_params
            self.logger.info(f"   冻结参数: {frozen_params:,}")
        
        # 显示模型结构
        self.logger.info(f"   模型结构: {self.model}")
    
    def _load_pretrained_weights(self):
        """加载预训练权重"""
        self.logger.info(f"📥 加载预训练模型: {self.args.pretrained_model}")
        
        try:
            # 加载检查点
            checkpoint = torch.load(self.args.pretrained_model, map_location=self.device)
            
            # 获取状态字典
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    self.logger.info(f"   检查点信息:")
                    if 'epoch' in checkpoint:
                        self.logger.info(f"     原始训练轮数: {checkpoint['epoch']}")
                    if 'best_val_loss' in checkpoint:
                        self.logger.info(f"     原始最佳验证损失: {checkpoint['best_val_loss']:.4f}")
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 智能加载权重
            loaded_keys, skipped_keys = self._smart_load_state_dict(state_dict)
            
            self.logger.info(f"✅ 成功加载 {len(loaded_keys)} 个权重")
            if skipped_keys:
                self.logger.info(f"⚠️ 跳过 {len(skipped_keys)} 个不兼容权重")
                if len(skipped_keys) <= 10:  # 只显示前10个
                    for key in skipped_keys[:10]:
                        self.logger.info(f"   - {key}")
            
        except Exception as e:
            self.logger.error(f"❌ 加载预训练权重失败: {e}")
            self.logger.warning(f"⚠️ 将从头开始训练")
    
    def _smart_load_state_dict(self, pretrained_dict):
        """智能加载状态字典"""
        model_dict = self.model.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for key, value in pretrained_dict.items():
            if key in model_dict:
                if model_dict[key].shape == value.shape:
                    model_dict[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {model_dict[key].shape} vs {value.shape})")
            else:
                skipped_keys.append(f"{key} (not found)")
        
        # 加载更新后的状态字典
        self.model.load_state_dict(model_dict)
        
        return loaded_keys, skipped_keys
    
    def _freeze_feature_layers(self):
        """冻结特征提取层"""
        self.logger.info("🔒 冻结特征提取层...")
        
        # 假设模型有features和classifier属性
        # 根据实际模型结构调整
        frozen_count = 0
        for name, param in self.model.named_parameters():
            # 冻结除了最后几层之外的所有层
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
                frozen_count += 1
        
        self.logger.info(f"   冻结了 {frozen_count} 个参数组")
        
        # 重新计算可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"   剩余可训练参数: {trainable_params:,}")
    
    def setup_training(self):
        """设置训练相关组件"""
        self.logger.info("=== 第3阶段：设置训练组件 ===")
        
        # 设置学习率
        base_lr = self.args.learning_rate
        if self.args.incremental_training and self.args.pretrained_model:
            # 增量训练使用更小的学习率
            effective_lr = base_lr * self.args.incremental_lr_factor
            self.logger.info(f"📉 增量训练调整学习率: {base_lr} → {effective_lr}")
        else:
            effective_lr = base_lr
            self.logger.info(f"🆕 标准训练学习率: {effective_lr}")
        
        # 设置优化器
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=effective_lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=effective_lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
        
        # 设置学习率调度器
        if self.args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.args.step_size, 
                gamma=self.args.gamma
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.args.epochs
            )
        else:
            self.scheduler = None
        
        # 设置损失函数 - 可选择是否使用浓度感知损失函数
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
        
        self.logger.info(f"训练组件设置完成:")
        self.logger.info(f"   优化器: {self.args.optimizer.upper()}")
        self.logger.info(f"   学习率: {effective_lr}")
        self.logger.info(f"   权重衰减: {self.args.weight_decay}")
        self.logger.info(f"   调度器: {self.args.scheduler}")
        if self.args.scheduler == 'step':
            self.logger.info(f"   StepLR参数: step_size={self.args.step_size}, gamma={self.args.gamma}")
        
        # 增量训练特殊设置
        if self.args.incremental_training:
            self.logger.info(f"   🔄 增量训练优化:")
            self.logger.info(f"     - 学习率因子: {self.args.incremental_lr_factor}")
            self.logger.info(f"     - 只训练可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
            # 增量训练可以使用更激进的早停
            if not hasattr(self.args, 'early_stopping_patience'):
                self.args.early_stopping_patience = 15  # 比正常训练更早停止
                self.logger.info(f"     - 早停耐心: {self.args.early_stopping_patience}")
        
        # 断点续训
        if self.args.resume:
            self.logger.info(f"从断点恢复训练: {self.args.resume}")
            self.load_checkpoint(self.args.resume)
    
    def save_checkpoint(self, epoch, train_losses, val_losses, val_maes, val_r2s, best_val_loss, best_epoch):
        """保存训练检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'val_maes': [float(x) for x in val_maes],
            'val_r2s': [float(x) for x in val_r2s],
            'best_val_loss': float(best_val_loss),
            'best_epoch': int(best_epoch),
            'args': self.args
        }
        
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"   检查点已保存: checkpoint_epoch_{epoch+1}.pth")
        
        # 保留最新检查点的软链接
        latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """加载训练检查点"""
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
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        
        # 检查是否是完整的检查点格式
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            # 完整检查点格式
            checkpoint = checkpoint_data
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("   ✓ 加载完整检查点格式")
        else:
            # 只有模型权重（best_model.pth格式）
            self.model.load_state_dict(checkpoint_data)
            self.logger.info("   ✓ 加载模型权重格式")
            # 创建一个最小的检查点格式返回
            checkpoint = {
                'epoch': 0,
                'model_state_dict': checkpoint_data,
                'optimizer_state_dict': None,
                'scheduler_state_dict': None,
                'train_losses': [],
                'val_losses': [],
                'val_maes': [],
                'val_r2s': [],
                'best_val_loss': float('inf'),
                'best_epoch': 0,
                'args': None
            }
            self.logger.warning("   ⚠️ 只有模型权重，其他状态将重新初始化")
            return checkpoint
        
        # 加载模型状态（已在上面处理）
        
        # 加载优化器状态（如果存在）
        if checkpoint.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"   优化器状态已恢复")
        else:
            self.logger.info(f"   优化器状态将重新初始化")
        
        # 加载调度器状态（如果存在）
        if self.scheduler and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info(f"   学习率调度器状态已恢复")
        else:
            if self.scheduler:
                self.logger.info(f"   学习率调度器状态将重新初始化")
        
        self.logger.info(f"检查点加载完成:")
        self.logger.info(f"   恢复到epoch: {checkpoint['epoch'] + 1}")
        self.logger.info(f"   历史最佳验证损失: {checkpoint['best_val_loss']:.6f}")
        self.logger.info(f"   历史最佳epoch: {checkpoint['best_epoch'] + 1}")
        
        return checkpoint
    
    def _generate_center_mask(self, size, device):
        """生成中心区域掩码，用于注意力约束"""
        h, w = size
        y, x = torch.meshgrid(torch.arange(h, device=device), 
                             torch.arange(w, device=device), indexing='ij')
        
        center_y, center_x = h // 2, w // 2
        distance = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_distance = torch.sqrt(torch.tensor(center_y ** 2 + center_x ** 2, device=device))
        
        # 高斯式衰减，中心权重最高
        mask = torch.exp(-0.5 * (distance / (max_distance * 0.3)) ** 2)
        return mask
    
    def _compute_attention_loss(self, features, weight=0.1):
        """计算注意力损失，鼓励模型关注中心区域"""
        if not hasattr(self.args, 'use_attention_constraint') or not self.args.use_attention_constraint:
            return 0.0
        
        # 计算特征图的平均激活
        activation_map = torch.mean(features, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 批量归一化激活图（避免原地操作）
        activation_map_flat = activation_map.view(activation_map.size(0), -1)  # [B, H*W]
        
        # 对每个样本进行归一化
        min_vals = activation_map_flat.min(dim=1, keepdim=True)[0]  # [B, 1]
        max_vals = activation_map_flat.max(dim=1, keepdim=True)[0]  # [B, 1]
        
        # 归一化到[0,1]，避免原地操作
        normalized_flat = (activation_map_flat - min_vals) / (max_vals - min_vals + 1e-8)
        activation_map_normalized = normalized_flat.view_as(activation_map)
        
        # 生成中心掩码
        center_mask = self._generate_center_mask(features.shape[-2:], features.device)
        center_mask = center_mask.unsqueeze(0).unsqueeze(0).expand_as(activation_map_normalized)
        
        # 计算注意力损失（鼓励激活图与中心掩码一致）
        attention_loss = nn.functional.mse_loss(activation_map_normalized, center_mask) * weight
        
        return attention_loss
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        # 更新渐进式权重损失函数的epoch
        if hasattr(self.criterion, 'update_epoch'):
            self.criterion.update_epoch(epoch)
        
        total_loss = 0.0
        total_mse_loss = 0.0
        total_attention_loss = 0.0
        num_batches = 0
        
        # 进度监控
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # 每10%输出一次
        
        for batch_idx, (images, targets, metadata) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs, features = self.model(images)  # 传统CNN返回 (concentration, features)
            
            # 主要回归损失
            mse_loss = self.criterion(outputs.squeeze(), targets.squeeze())
            
            # 注意力约束损失
            attention_loss = self._compute_attention_loss(features, 
                                                        weight=getattr(self.args, 'attention_weight', 0.1))
            
            # 总损失
            total_batch_loss = mse_loss + attention_loss
            
            # 反向传播
            total_batch_loss.backward()
            # 添加梯度裁剪，提高训练稳定性
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_mse_loss += mse_loss.item()
            total_attention_loss += attention_loss if isinstance(attention_loss, (int, float)) else attention_loss.item()
            num_batches += 1
            
            # 详细进度输出
            if batch_idx % log_interval == 0:
                progress = (batch_idx + 1) / total_batches * 100
                avg_loss = total_loss / num_batches
                avg_mse = total_mse_loss / num_batches
                avg_attention = total_attention_loss / num_batches
                
                if hasattr(self.args, 'use_attention_constraint') and self.args.use_attention_constraint:
                    self.logger.info(f"   训练进度: {batch_idx+1}/{total_batches} ({progress:.1f}%) - "
                                   f"总损失: {total_batch_loss.item():.6f} "
                                   f"(MSE: {mse_loss.item():.6f}, 注意力: {avg_attention:.6f})")
                else:
                    self.logger.info(f"   训练进度: {batch_idx+1}/{total_batches} ({progress:.1f}%) - "
                                   f"当前损失: {mse_loss.item():.6f}, 平均损失: {avg_loss:.6f}")
        
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
                
                outputs, _ = self.model(images)  # 传统CNN返回 (concentration, features)
                loss = self.criterion(outputs.squeeze(), batch_targets.squeeze())
                
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
            
            # 训练
            self.logger.info("训练阶段:")
            train_loss = self.train_epoch(epoch)
            
            # 验证
            self.logger.info("验证阶段:")
            val_loss, val_mse, val_mae, val_r2, predictions, targets = self.validate()
            
            # 更新学习率
            if self.scheduler:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    self.logger.info(f"   学习率更新: {old_lr:.6f} -> {new_lr:.6f}")
            
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
                torch.save(self.model.state_dict(), 
                          os.path.join(self.output_dir, 'best_model.pth'))
                self.logger.info(f"   ✓ 新的最佳模型 (验证损失: {val_loss:.6f})")
            else:
                improvement = val_loss - best_val_loss
                self.logger.info(f"   当前模型比最佳模型差 {improvement:.6f}")
            
            # 保存检查点
            if (epoch + 1) % self.args.save_checkpoint_every == 0:
                self.save_checkpoint(epoch, train_losses, val_losses, val_maes, val_r2s, best_val_loss, best_epoch)
            
            # 每5个epoch生成可视化
            if (epoch + 1) % 5 == 0:
                try:
                    labels = get_labels(CHINESE_SUPPORTED)
                    scatter_path = os.path.join(self.output_dir, f'predictions_epoch_{epoch+1}.png')
                    title = f"{labels['epoch']} {epoch+1}: {labels['prediction_vs_true']} (R²={val_r2:.4f})"
                    create_prediction_scatter_plot(targets, predictions, scatter_path, title=title)
                    self.logger.info(f"   预测可视化已保存: predictions_epoch_{epoch+1}.png")
                except Exception as e:
                    self.logger.warning(f"生成预测可视化失败: {e}")
        
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
        
        # 最终预测vs真实值
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, 'best_model.pth')))
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

    def train_epoch_with_dual_eval(self, epoch):
        """训练一个epoch，同时跟踪平滑权重损失和标准MSE"""
        self.model.train()
        
        # 更新渐进式权重
        if hasattr(self.criterion, 'update_epoch'):
            self.criterion.update_epoch(epoch)
        
        total_smooth_loss = 0.0
        total_standard_loss = 0.0
        total_mse_loss = 0.0
        total_attention_loss = 0.0
        num_batches = 0
        
        # 标准MSE用于性能评估
        standard_mse = nn.MSELoss()
        
        # 进度监控
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # 每10%输出一次
        
        for batch_idx, (images, targets, metadata) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device).float()
            
            self.optimizer.zero_grad()
            outputs, features = self.model(images)  # 传统CNN返回 (concentration, features)
            
            # 处理模型输出
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 用于训练的平滑权重损失
            smooth_loss = self.criterion(outputs.squeeze(), targets.squeeze())
            
            # 计算注意力约束损失
            attention_loss = 0.0
            if self.args.use_attention_constraint and features is not None:
                attention_loss = self._compute_attention_loss(features, self.args.attention_weight)
                smooth_loss = smooth_loss + attention_loss
            
            # 用于评估的标准MSE（不参与梯度更新）
            with torch.no_grad():
                standard_loss = standard_mse(outputs.squeeze(), targets.squeeze())
            
            smooth_loss.backward()
            self.optimizer.step()
            
            total_smooth_loss += smooth_loss.item()
            total_standard_loss += standard_loss.item()
            
            if attention_loss:
                total_attention_loss += attention_loss
            
            num_batches += 1
            
            # 进度输出
            if batch_idx % log_interval == 0:
                progress = 100.0 * batch_idx / total_batches
                self.logger.info(f"   批次 {batch_idx:3d}/{total_batches} ({progress:5.1f}%) | "
                               f"平滑损失: {smooth_loss.item():.4f} | "
                               f"标准MSE: {standard_loss.item():.4f}")
        
        avg_smooth_loss = total_smooth_loss / num_batches
        avg_standard_loss = total_standard_loss / num_batches
        avg_attention_loss = total_attention_loss / num_batches if total_attention_loss > 0 else 0.0
        
        return avg_smooth_loss, avg_standard_loss, avg_attention_loss

    def validate_with_dual_eval(self):
        """验证模型，返回平滑权重损失和标准MSE"""
        self.model.eval()
        total_smooth_loss = 0.0
        total_standard_loss = 0.0
        total_attention_loss = 0.0
        predictions = []
        targets = []
        
        standard_mse = nn.MSELoss()
        
        with torch.no_grad():
            for images, batch_targets, metadata in self.val_loader:
                images = images.to(self.device)
                batch_targets = batch_targets.to(self.device).float()
                
                outputs, features = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # 平滑权重损失
                smooth_loss = self.criterion(outputs.squeeze(), batch_targets.squeeze())
                
                # 计算注意力约束损失
                attention_loss = 0.0
                if self.args.use_attention_constraint and features is not None:
                    attention_loss = self._compute_attention_loss(features, self.args.attention_weight)
                    smooth_loss = smooth_loss + attention_loss
                
                # 标准MSE
                standard_loss = standard_mse(outputs.squeeze(), batch_targets.squeeze())
                
                total_smooth_loss += smooth_loss.item()
                total_standard_loss += standard_loss.item()
                
                if attention_loss:
                    total_attention_loss += attention_loss
                
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        avg_smooth_loss = total_smooth_loss / len(self.val_loader)
        avg_standard_loss = total_standard_loss / len(self.val_loader)
        avg_attention_loss = total_attention_loss / len(self.val_loader) if total_attention_loss > 0 else 0.0
        
        # 计算其他指标
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return (avg_smooth_loss, avg_standard_loss, avg_attention_loss, 
                mae, r2, predictions, targets)

def main():
    parser = argparse.ArgumentParser(description='传统CNN在特征数据集上的训练')
    
    # 数据集参数
    parser.add_argument('--feature_dataset_path', type=str, default=None,
                       help='特征数据集路径 (默认自动检测最新的)')
    parser.add_argument('--bg_mode', type=str, default='all', 
                       choices=['bg0', 'bg1', 'all', 'bg0_20mw', 'bg0_100mw', 'bg0_400mw', 
                               'bg1_20mw', 'bg1_100mw', 'bg1_400mw'],
                       help='训练模式: bg0/bg1/all(传统3档) 或 bg0_20mw等(新6档细分)')
    parser.add_argument('--power_filter', type=str, default=None,
                       help='过滤特定功率 (例如: 20mw, 100mw, 400mw) - 与bg_mode细分冲突时忽略')
    parser.add_argument('--concentration_range', type=str, default=None,
                       help='浓度范围过滤 (格式: "min,max" 例如: "0,500")')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--split_method', type=str, default='random',
                       choices=['random', 'stratified', 'interval'],
                       help='数据集划分方式: random(随机), stratified(按浓度分层), interval(按浓度间隔)')
    parser.add_argument('--train_interval_length', type=int, default=4,
                       help='训练集浓度间隔长度 (仅在split_method=interval时有效)')
    parser.add_argument('--val_interval_length', type=int, default=1,
                       help='验证集浓度间隔长度 (仅在split_method=interval时有效)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='优化器')
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'none'],
                       help='学习率调度器')
    parser.add_argument('--step_size', type=int, default=20,
                       help='StepLR步长')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='StepLR衰减率')
    
    # 断点续训参数
    parser.add_argument('--resume', type=str, default=None,
                       help='从指定检查点恢复训练（输出目录路径）')
    parser.add_argument('--save_checkpoint_every', type=int, default=10,
                       help='每N个epoch保存一次检查点 (默认: 10)')
    
    # 增量训练参数
    parser.add_argument('--incremental_training', action='store_true',
                       help='启用增量训练模式')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='预训练模型路径 (用于增量训练)')
    parser.add_argument('--incremental_lr_factor', type=float, default=0.1,
                       help='增量训练学习率因子 (默认: 0.1)')
    parser.add_argument('--freeze_features', action='store_true',
                       help='冻结特征提取层，仅训练分类器')
    
    # 注意力约束参数（抑制捷径学习）
    parser.add_argument('--use_attention_constraint', action='store_true',
                       help='启用注意力约束，鼓励模型关注中心区域')
    parser.add_argument('--attention_weight', type=float, default=0.1,
                       help='注意力损失权重 (默认: 0.1)')
    
    # 高浓度优化参数
    parser.add_argument('--use_concentration_aware_loss', action='store_true',
                       help='启用浓度感知损失函数 (默认启用)')
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
    
    print("传统CNN特征数据集训练 (支持6档细分 + 增量训练 + 注意力约束 + 浓度感知损失)")
    print("=" * 70)
    print(f"训练模式: {args.bg_mode.upper()}")
    
    if '_' in args.bg_mode:
        parts = args.bg_mode.split('_')
        print(f"   >> 6档细分模式: {parts[0]} + {parts[1]}")
    else:
        print(f"   >> 传统3档模式: {args.bg_mode}")
        if args.power_filter:
            print(f"   >> 功率过滤: {args.power_filter}")
    
    # 显示增量训练配置
    if args.incremental_training:
        print(f"   >> 增量训练: 启用")
        print(f"      - 预训练模型: {args.pretrained_model if args.pretrained_model else '未指定'}")
        print(f"      - 学习率因子: {args.incremental_lr_factor}")
        print(f"      - 冻结特征层: {'是' if args.freeze_features else '否'}")
        
        # 验证增量训练参数
        if args.pretrained_model and not os.path.exists(args.pretrained_model):
            print(f"   ⚠️  警告: 预训练模型文件不存在: {args.pretrained_model}")
            print(f"      将从头开始训练")
    else:
        print(f"   >> 增量训练: 禁用 (使用 --incremental_training 启用)")
    
    # 显示注意力约束配置
    if args.use_attention_constraint:
        print(f"   >> 注意力约束: 启用 (权重: {args.attention_weight})")
        print("      - 鼓励模型关注中心区域，抑制捷径学习")
    else:
        print(f"   >> 注意力约束: 禁用 (使用 --use_attention_constraint 启用)")
    
    # 显示浓度感知损失函数配置
    if not getattr(args, 'disable_concentration_aware_loss', False):
        threshold = getattr(args, 'high_concentration_threshold', 800)
        weight = getattr(args, 'high_concentration_weight', 0.5)
        print(f"   >> 浓度感知损失: 启用 (阈值: {threshold}mg/L, 权重: {weight})")
        print(f"      - 高浓度区域降权{(1-weight)*100:.0f}%, 缓解特征趋同问题")
    else:
        print(f"   >> 浓度感知损失: 禁用 (使用标准MSE损失)")
    
    try:
        # 创建训练器
        trainer = BaselineCNNTrainer(args)
        
        # 加载数据
        dataset = trainer.load_data()
        
        # 创建模型
        trainer.create_model()
        
        # 设置训练
        trainer.setup_training()
        
        # 训练模型
        history = trainer.train()
        
        print(f"\n训练完成！")
        print(f"结果保存在: {trainer.output_dir}")
        print(f"最佳模型: {trainer.output_dir}/best_model.pth")
        
        # 增量训练总结
        if args.incremental_training:
            print(f"\n📊 增量训练总结:")
            print(f"   原始模型: {args.pretrained_model if args.pretrained_model else '无'}")
            print(f"   学习率调整: {args.learning_rate} × {args.incremental_lr_factor} = {args.learning_rate * args.incremental_lr_factor}")
            print(f"   特征层冻结: {'是' if args.freeze_features else '否'}")
            print(f"   最终验证损失: {history['best_val_loss']:.6f}")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()