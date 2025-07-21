#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征数据集加载器
专门用于加载YOLO裁剪后的特征数据集
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from typing import List, Tuple, Optional


class FeatureImageDataset(Dataset):
    """特征图像数据集加载器"""
    
    def __init__(self, 
                 feature_dataset_path: str,
                 transform=None,
                 bg_type: Optional[str] = None,
                 power_filter: Optional[str] = None):
        """
        初始化特征数据集
        
        Args:
            feature_dataset_path: 特征数据集路径
            transform: 图像变换
            bg_type: 过滤特定背景类型 ('bg0', 'bg1')
            power_filter: 过滤特定功率 ('20mw', '100mw', '400mw')
        """
        self.feature_dataset_path = feature_dataset_path
        self.transform = transform
        self.bg_type = bg_type
        self.power_filter = power_filter
        
        # 特征图像和信息文件路径
        self.images_dir = os.path.join(feature_dataset_path, 'images')
        self.info_dir = os.path.join(feature_dataset_path, 'original_info')
        
        # 检查路径
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"特征图像目录不存在: {self.images_dir}")
        if not os.path.exists(self.info_dir):
            raise FileNotFoundError(f"信息文件目录不存在: {self.info_dir}")
        
        # 加载数据
        self.image_files = []
        self.concentrations = []
        self.metadata = []
        
        self._load_dataset()
        
        print(f"特征数据集加载完成")
        print(f"   数据集路径: {feature_dataset_path}")
        print(f"   有效样本: {len(self.image_files)} 张")
        if bg_type:
            print(f"   背景过滤: {bg_type}")
        if power_filter:
            print(f"   功率过滤: {power_filter}")
        if len(self.concentrations) > 0:
            print(f"   浓度范围: {min(self.concentrations):.1f} - {max(self.concentrations):.1f}")
            print(f"   平均浓度: {np.mean(self.concentrations):.2f}")
    
    def _load_dataset(self):
        """加载数据集"""
        print(f"\n=== 开始加载特征数据集 ===")
        print(f"图像目录: {self.images_dir}")
        print(f"信息目录: {self.info_dir}")
        
        # 获取所有特征图像文件
        print("步骤1/4: 扫描特征图像文件...")
        image_pattern = os.path.join(self.images_dir, "feature_*.jpg")
        image_files = glob.glob(image_pattern)
        
        print(f"找到 {len(image_files)} 个特征图像文件")
        
        # 添加过滤信息
        filter_info = []
        if self.bg_type:
            filter_info.append(f"背景类型={self.bg_type}")
        if self.power_filter:
            filter_info.append(f"功率={self.power_filter}")
        
        if filter_info:
            print(f"应用过滤条件: {', '.join(filter_info)}")
        else:
            print("无过滤条件，加载全部数据")
        
        print("\n步骤2/4: 验证图像和信息文件...")
        valid_count = 0
        invalid_count = 0
        filtered_count = 0
        
        # 计算进度输出间隔
        total_files = len(image_files)
        progress_interval = max(1, total_files // 20)  # 每5%输出一次进度
        
        for i, image_file in enumerate(image_files):
            # 进度输出
            if i % progress_interval == 0 or i == total_files - 1:
                progress = (i + 1) / total_files * 100
                print(f"  进度: {i+1}/{total_files} ({progress:.1f}%) - 有效:{valid_count}, 过滤:{filtered_count}, 无效:{invalid_count}")
            
            try:
                # 构建对应的信息文件路径
                image_name = os.path.basename(image_file)
                info_name = os.path.splitext(image_name)[0] + '.json'
                info_file = os.path.join(self.info_dir, info_name)
                
                if not os.path.exists(info_file):
                    if invalid_count < 5:  # 只显示前5个错误，避免刷屏
                        print(f"  警告: 信息文件不存在: {info_file}")
                    invalid_count += 1
                    continue
                
                # 加载信息文件
                with open(info_file, 'r', encoding='utf-8') as f:
                    info_data = json.load(f)
                
                # 检查过滤条件
                if self.bg_type and info_data.get('bg_type') != self.bg_type:
                    filtered_count += 1
                    continue
                
                if self.power_filter and info_data.get('power') != self.power_filter:
                    filtered_count += 1
                    continue
                
                # 验证图像文件
                if not self._validate_image(image_file):
                    invalid_count += 1
                    continue
                
                # 添加到数据集
                self.image_files.append(image_file)
                self.concentrations.append(float(info_data['concentration']))
                self.metadata.append(info_data)
                valid_count += 1
                
            except Exception as e:
                if invalid_count < 5:  # 只显示前5个错误
                    print(f"  加载文件失败: {image_file} - {e}")
                invalid_count += 1
        
        print(f"\n步骤3/4: 数据集加载统计")
        print(f"  总文件数: {total_files}")
        print(f"  有效文件: {valid_count}")
        print(f"  过滤文件: {filtered_count}")
        print(f"  无效文件: {invalid_count}")
        print(f"  最终数据集大小: {valid_count} 张图像")
        
        if len(self.image_files) == 0:
            raise ValueError("未找到有效的特征图像数据！")
        
        print("\n步骤4/4: 计算数据集统计信息...")
        if len(self.concentrations) > 0:
            concentrations = np.array(self.concentrations)
            print(f"  浓度范围: {concentrations.min():.1f} - {concentrations.max():.1f}")
            print(f"  平均浓度: {concentrations.mean():.2f} ± {concentrations.std():.2f}")
            print(f"  浓度种类: {len(set(self.concentrations))} 种")
        
        print("=== 数据集加载完成 ===\n")
    
    def _validate_image(self, image_path: str) -> bool:
        """验证图像文件完整性"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """获取数据项"""
        image_path = self.image_files[idx]
        concentration = self.concentrations[idx]
        metadata = self.metadata[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, concentration, metadata
            
        except Exception as e:
            print(f"读取图像失败: {image_path} - {e}")
            # 返回一个占位图像
            if self.transform:
                placeholder = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                placeholder = Image.new('RGB', (224, 224), color='black')
            return placeholder, concentration, metadata
    
    def get_concentration_statistics(self):
        """获取浓度统计信息"""
        if len(self.concentrations) == 0:
            return {}
        
        concentrations = np.array(self.concentrations)
        return {
            'count': len(concentrations),
            'min': float(concentrations.min()),
            'max': float(concentrations.max()),
            'mean': float(concentrations.mean()),
            'std': float(concentrations.std()),
            'unique_values': sorted(list(set(concentrations)))
        }
    
    def get_metadata_statistics(self):
        """获取元数据统计信息"""
        if len(self.metadata) == 0:
            return {}
        
        stats = {}
        
        # 统计背景类型
        bg_types = [meta['bg_type'] for meta in self.metadata]
        stats['bg_types'] = {bg: bg_types.count(bg) for bg in set(bg_types)}
        
        # 统计功率类型
        powers = [meta['power'] for meta in self.metadata]
        stats['powers'] = {power: powers.count(power) for power in set(powers)}
        
        # 统计距离
        distances = [meta['distance'] for meta in self.metadata]
        stats['distances'] = {dist: distances.count(dist) for dist in set(distances)}
        
        # 统计检测置信度
        confidences = [meta['detection_confidence'] for meta in self.metadata]
        stats['detection_confidence'] = {
            'min': min(confidences),
            'max': max(confidences),
            'mean': np.mean(confidences),
            'std': np.std(confidences)
        }
        
        return stats


def create_feature_dataloader(feature_dataset_path: str,
                             batch_size: int = 32,
                             shuffle: bool = True,
                             bg_type: Optional[str] = None,
                             power_filter: Optional[str] = None,
                             image_size: int = 224) -> Tuple[DataLoader, FeatureImageDataset]:
    """
    创建特征数据集的DataLoader
    
    Args:
        feature_dataset_path: 特征数据集路径
        batch_size: 批次大小
        shuffle: 是否随机打乱
        bg_type: 背景类型过滤 ('bg0', 'bg1')
        power_filter: 功率过滤 ('20mw', '100mw', '400mw')
        image_size: 图像尺寸
    
    Returns:
        DataLoader 和 Dataset 对象
    """
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = FeatureImageDataset(
        feature_dataset_path=feature_dataset_path,
        transform=transform,
        bg_type=bg_type,
        power_filter=power_filter
    )
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader, dataset


def detect_feature_datasets(base_path: str = ".") -> List[str]:
    """
    检测当前目录下的特征数据集
    
    Args:
        base_path: 搜索基础路径
    
    Returns:
        特征数据集路径列表
    """
    feature_datasets = []
    
    # 1. 搜索feature_dataset_*目录（带时间戳）
    pattern = os.path.join(base_path, "feature_dataset_*")
    potential_dirs = glob.glob(pattern)
    
    for dir_path in potential_dirs:
        if os.path.isdir(dir_path):
            # 检查是否包含必要的子目录
            images_dir = os.path.join(dir_path, 'images')
            info_dir = os.path.join(dir_path, 'original_info')
            dataset_info = os.path.join(dir_path, 'dataset_info.json')
            
            if all(os.path.exists(p) for p in [images_dir, info_dir, dataset_info]):
                feature_datasets.append(dir_path)
    
    # 2. 搜索feature_dataset目录（不带时间戳）
    simple_feature_dataset = os.path.join(base_path, "feature_dataset")
    if os.path.isdir(simple_feature_dataset):
        # 检查是否包含必要的子目录
        images_dir = os.path.join(simple_feature_dataset, 'images')
        info_dir = os.path.join(simple_feature_dataset, 'original_info')
        dataset_info = os.path.join(simple_feature_dataset, 'dataset_info.json')
        
        if all(os.path.exists(p) for p in [images_dir, info_dir, dataset_info]):
            feature_datasets.append(simple_feature_dataset)
    
    return sorted(feature_datasets)


def main():
    """测试特征数据集加载器"""
    print("测试特征数据集加载器")
    
    # 检测特征数据集
    feature_datasets = detect_feature_datasets()
    
    if not feature_datasets:
        print("未找到特征数据集")
        print("请先运行 create_feature_dataset.py 生成特征数据集")
        return
    
    # 使用最新的特征数据集
    latest_dataset = feature_datasets[-1]
    print(f"使用数据集: {latest_dataset}")
    
    try:
        # 创建DataLoader
        dataloader, dataset = create_feature_dataloader(
            feature_dataset_path=latest_dataset,
            batch_size=16,
            bg_type=None  # 不过滤
        )
        
        print(f"\n数据集统计:")
        print(f"   总样本数: {len(dataset)}")
        
        # 浓度统计
        conc_stats = dataset.get_concentration_statistics()
        print(f"   浓度范围: {conc_stats['min']:.1f} - {conc_stats['max']:.1f}")
        print(f"   平均浓度: {conc_stats['mean']:.2f} ± {conc_stats['std']:.2f}")
        print(f"   浓度种类: {len(conc_stats['unique_values'])} 种")
        
        # 元数据统计
        meta_stats = dataset.get_metadata_statistics()
        print(f"   背景分布: {meta_stats['bg_types']}")
        print(f"   功率分布: {meta_stats['powers']}")
        print(f"   平均置信度: {meta_stats['detection_confidence']['mean']:.3f}")
        
        # 测试数据加载
        print(f"\n测试数据加载...")
        for i, (images, concentrations, metadata) in enumerate(dataloader):
            print(f"   批次 {i+1}: {images.shape}, 浓度范围: {concentrations.min():.1f}-{concentrations.max():.1f}")
            if i >= 2:  # 只测试前3个批次
                break
        
        print(f"特征数据集加载器测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    main() 