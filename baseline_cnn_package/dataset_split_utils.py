"""数据集划分工具模块

提供多种数据集划分方式：
1. 随机划分（默认）
2. 按浓度分层抽样（每个浓度8:2）
3. 按浓度间隔分配（每4个连续浓度为训练集，随后1个为验证集）
"""

import torch
from torch.utils.data import random_split, Subset
import numpy as np


def split_dataset_by_random(dataset, train_ratio=0.8, seed=42):
    """
    随机划分数据集
    
    Args:
        dataset: 数据集对象
        train_ratio: 训练集比例
        seed: 随机种子
    
    Returns:
        train_dataset, val_dataset: 训练集和验证集
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset


def split_dataset_by_concentration_stratified(dataset, train_ratio=0.8, seed=42):
    """
    按浓度分层抽样划分数据集
    每个浓度的图像均按指定比例随机分配至训练集与验证集
    
    Args:
        dataset: 数据集对象
        train_ratio: 训练集比例
        seed: 随机种子
    
    Returns:
        train_dataset, val_dataset: 训练集和验证集
    """
    # 获取所有样本的浓度
    concentrations = []
    for i in range(len(dataset)):
        # 假设dataset[i]返回(image, concentration, metadata)
        _, concentration, _ = dataset[i]
        concentrations.append(concentration)
    
    concentrations = np.array(concentrations)
    unique_concentrations = np.unique(concentrations)
    
    # 为每个浓度划分样本
    train_indices = []
    val_indices = []
    
    generator = np.random.default_rng(seed)
    
    for conc in unique_concentrations:
        # 获取该浓度的所有样本索引
        conc_indices = np.where(concentrations == conc)[0]
        
        # 随机打乱
        generator.shuffle(conc_indices)
        
        # 按比例分割
        train_size = int(len(conc_indices) * train_ratio)
        train_indices.extend(conc_indices[:train_size])
        val_indices.extend(conc_indices[train_size:])
    
    # 创建子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def split_dataset_by_concentration_interval(dataset, train_interval_length=4, val_interval_length=1, seed=42):
    """
    按浓度间隔分配划分数据集（循环分配模式）
    每train_interval_length个连续浓度为训练集，随后val_interval_length个为验证集，如此循环
    例如：train_interval_length=4, val_interval_length=1时，
    浓度0-3进入训练集，浓度4进入验证集，浓度5-8进入训练集，浓度9进入验证集，以此类推
    
    Args:
        dataset: 数据集对象
        train_interval_length: 训练集浓度间隔长度
        val_interval_length: 验证集浓度间隔长度
        seed: 随机种子（用于打乱样本顺序）
    
    Returns:
        train_dataset, val_dataset: 训练集和验证集
    """
    # 获取所有样本的浓度
    concentrations = []
    for i in range(len(dataset)):
        # 假设dataset[i]返回(image, concentration, metadata)
        _, concentration, _ = dataset[i]
        concentrations.append(concentration)
    
    concentrations = np.array(concentrations)
    unique_concentrations = np.unique(concentrations)
    
    # 按浓度排序
    unique_concentrations = np.sort(unique_concentrations)
    
    # 使用循环分配逻辑
    train_concentrations = []
    val_concentrations = []
    
    cycle_length = train_interval_length + val_interval_length
    
    for i, conc in enumerate(unique_concentrations):
        position_in_cycle = i % cycle_length
        if position_in_cycle < train_interval_length:
            train_concentrations.append(conc)
        else:
            val_concentrations.append(conc)
    
    # 获取对应的样本索引
    train_indices = []
    val_indices = []
    
    for i in range(len(dataset)):
        conc = concentrations[i]
        if conc in train_concentrations:
            train_indices.append(i)
        elif conc in val_concentrations:
            val_indices.append(i)
    
    # 随机打乱样本顺序
    generator = np.random.default_rng(seed)
    generator.shuffle(train_indices)
    generator.shuffle(val_indices)
    
    # 创建子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def split_dataset(dataset, split_method='random', train_ratio=0.8, train_interval_length=4, val_interval_length=1, seed=42):
    """
    通用数据集划分函数
    
    Args:
        dataset: 数据集对象
        split_method: 划分方法 ('random', 'stratified', 'interval')
        train_ratio: 训练集比例（用于random和stratified方法）
        train_interval_length: 训练集浓度间隔长度（用于interval方法）
        val_interval_length: 验证集浓度间隔长度（用于interval方法）
        seed: 随机种子
    
    Returns:
        train_dataset, val_dataset: 训练集和验证集
    """
    if split_method == 'random':
        return split_dataset_by_random(dataset, train_ratio, seed)
    elif split_method == 'stratified':
        return split_dataset_by_concentration_stratified(dataset, train_ratio, seed)
    elif split_method == 'interval':
        return split_dataset_by_concentration_interval(dataset, train_interval_length, val_interval_length, seed)
    else:
        raise ValueError(f"Unsupported split method: {split_method}")