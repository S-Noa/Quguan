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
from typing import List, Tuple, Optional, Dict, Callable
import re

class FeatureImageDataset(Dataset):
    """特征图像数据集加载器"""
    
    def __init__(self, 
                 feature_dataset_path: str,
                 transform=None,
                 bg_type: Optional[str] = None,
                 power_filter: Optional[str] = None,
                 concentration_range: Optional[Tuple[float, float]] = None):
        """
        初始化特征数据集
        
        Args:
            feature_dataset_path: 特征数据集路径
            transform: 图像变换
            bg_type: 过滤特定背景类型 ('bg0', 'bg1')
            power_filter: 过滤特定功率 ('20mw', '100mw', '400mw')
            concentration_range: 浓度过滤范围 (min, max)
        """
        self.feature_dataset_path = feature_dataset_path
        self.transform = transform
        self.bg_type = bg_type
        self.power_filter = power_filter
        self.concentration_range = concentration_range
        
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
        # 兼容本地和云端两种命名模式
        image_pattern1 = os.path.join(self.images_dir, "*_cropped.jpg")
        image_pattern2 = os.path.join(self.images_dir, "feature_*.jpg")
        image_files = glob.glob(image_pattern1) + glob.glob(image_pattern2)
        
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
                # 根据不同命名模式构建信息文件名
                if image_name.endswith('_cropped.jpg'):
                    # 云端命名模式: xxx_cropped.jpg -> xxx_metadata.json
                    info_name = image_name[:-len('_cropped.jpg')] + '_metadata.json'
                elif image_name.startswith('feature_') and image_name.endswith('.jpg'):
                    # 本地命名模式: feature_xxx.jpg -> feature_xxx.json
                    info_name = image_name[:-len('.jpg')] + '.json'
                else:
                    # 其他模式: xxx.jpg -> xxx.json
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
                # 首先尝试从info_data获取背景类型和功率信息
                bg_type = info_data.get('bg_type')
                power = info_data.get('power')
                
                # 如果info_data中没有这些信息，则从文件名解析
                if bg_type is None or power is None:
                    # 从文件名解析信息
                    # 文件名格式: 
                    # 云端模式: 入射角度-悬浮物浓度-相机高度-水体流速-背景补光与否-激光光强-序号_cropped.jpg
                    # 本地模式: feature_入射角度-悬浮物浓度-相机高度-水体流速-背景补光与否-激光光强.jpg
                    filename = os.path.basename(image_file)
                    # 处理文件名中的特殊字符，包括'_cropped'后缀
                    name_without_ext = os.path.splitext(filename)[0]
                    # 移除'_cropped'后缀（仅对云端模式）
                    if name_without_ext.endswith('_cropped'):
                        name_without_ext = name_without_ext[:-len('_cropped')]
                    # 移除'feature_'前缀（仅对本地模式）
                    if name_without_ext.startswith('feature_'):
                        name_without_ext = name_without_ext[len('feature_'):]
                    parts = name_without_ext.split('-')
                    if len(parts) >= 6:
                        try:
                            if bg_type is None:
                                bg_type = parts[4]  # 背景补光与否是第五个部分
                            if power is None:
                                power = parts[5]  # 激光光强是第六个部分
                        except (IndexError) as e:
                            # 如果解析失败，保持为None
                            pass
                
                if self.bg_type and bg_type != self.bg_type:
                    filtered_count += 1
                    continue
                
                if self.power_filter and power != self.power_filter:
                    filtered_count += 1
                    continue
                
                # 检查浓度范围过滤条件
                if self.concentration_range:
                    min_conc, max_conc = self.concentration_range
                    concentration = info_data.get('concentration')
                    # 如果没有浓度信息，尝试从文件名解析
                    if concentration is None:
                        # 从文件名解析信息
                        # 文件名格式: 
                        # 云端模式: 入射角度-悬浮物浓度-相机高度-水体流速-背景补光与否-激光光强-序号_cropped.jpg
                        # 本地模式: feature_入射角度-悬浮物浓度-相机高度-水体流速-背景补光与否-激光光强.jpg
                        filename = os.path.basename(image_file)
                        # 处理文件名中的特殊字符，包括'_cropped'或'_metadata'后缀
                        name_without_ext = os.path.splitext(filename)[0]
                        # 移除'_cropped'后缀（仅对云端模式）
                        if name_without_ext.endswith('_cropped'):
                            name_without_ext = name_without_ext[:-len('_cropped')]
                        # 移除'_metadata'后缀（仅对云端模式）
                        elif name_without_ext.endswith('_metadata'):
                            name_without_ext = name_without_ext[:-len('_metadata')]
                        # 移除'feature_'前缀（仅对本地模式）
                        if name_without_ext.startswith('feature_'):
                            name_without_ext = name_without_ext[len('feature_'):]
                        parts = name_without_ext.split('-')
                        if len(parts) >= 6:
                            try:
                                concentration = float(parts[1])  # 悬浮物浓度是第二个部分
                            except (ValueError, IndexError) as e:
                                # 如果解析失败，使用默认值0.0
                                print(f"  警告: 浓度范围过滤时文件名解析失败 {filename}: {e}")
                                concentration = 0.0
                        else:
                            # 如果文件名格式不正确，使用默认值0.0
                            print(f"  警告: 浓度范围过滤时文件名格式不正确 {filename}: 部分数 {len(parts)} < 6")
                            concentration = 0.0
                    else:
                        concentration = float(concentration)
                
                    if concentration < min_conc or concentration > max_conc:
                        filtered_count += 1
                        continue
                
                # 验证图像文件
                if not self._validate_image(image_file):
                    invalid_count += 1
                    continue
                
                # 添加到数据集
                self.image_files.append(image_file)
                # 处理可能缺失的浓度信息
                concentration = info_data.get('concentration')
                bg_type = info_data.get('bg_type')
                power = info_data.get('power')
                
                # 如果没有浓度信息，尝试从文件名解析
                if concentration is None:
                    # 从文件名解析信息
                    # 文件名格式: 
                    # 云端模式: 入射角度-悬浮物浓度-相机高度-水体流速-背景补光与否-激光光强-序号_cropped.jpg
                    # 本地模式: feature_入射角度-悬浮物浓度-相机高度-水体流速-背景补光与否-激光光强.jpg
                    filename = os.path.basename(image_file)
                    # 处理文件名中的特殊字符，包括'_cropped'或'_metadata'后缀
                    name_without_ext = os.path.splitext(filename)[0]
                    # 移除'_cropped'后缀（仅对云端模式）
                    if name_without_ext.endswith('_cropped'):
                        name_without_ext = name_without_ext[:-len('_cropped')]
                    # 移除'_metadata'后缀（仅对云端模式）
                    elif name_without_ext.endswith('_metadata'):
                        name_without_ext = name_without_ext[:-len('_metadata')]
                    # 移除'feature_'前缀（仅对本地模式）
                    if name_without_ext.startswith('feature_'):
                        name_without_ext = name_without_ext[len('feature_'):]
                    parts = name_without_ext.split('-')
                    if len(parts) >= 6:
                        try:
                            concentration = float(parts[1])  # 悬浮物浓度是第二个部分
                            # 更新元数据
                            info_data['concentration'] = concentration
                            # 如果背景类型缺失，也从文件名解析
                            if bg_type is None:
                                bg_type = parts[4]  # 背景补光与否是第五个部分
                                info_data['bg_type'] = bg_type
                            # 如果功率缺失，也从文件名解析
                            if power is None:
                                power = parts[5]  # 激光光强是第六个部分
                                info_data['power'] = power
                        except (ValueError, IndexError) as e:
                            # 如果解析失败，使用默认值0.0
                            print(f"  警告: 文件名浓度解析失败 {filename}: {e}")
                            concentration = 0.0
                    else:
                        # 如果文件名格式不正确，使用默认值0.0
                        print(f"  警告: 文件名格式不正确 {filename}: 部分数 {len(parts)} < 6")
                        concentration = 0.0
                else:
                    concentration = float(concentration)
                
                self.concentrations.append(concentration)
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
        metadata = self.metadata[idx].copy()  # 创建副本以避免修改原始数据
        
        # 处理元数据字段，确保兼容性
        # 如果存在'bbox'字段，将其重命名为'detection_bbox'
        if 'bbox' in metadata and 'detection_bbox' not in metadata:
            metadata['detection_bbox'] = metadata['bbox']
        
        # 如果不存在'detection_confidence'字段，设置默认值
        if 'detection_confidence' not in metadata:
            metadata['detection_confidence'] = 1.0
        
        # 添加调试信息
        if idx < 5:  # 只显示前5个样本的详细信息
            print(f"调试信息 - 样本 {idx}: {os.path.basename(image_path)}")
            print(f"  浓度值: {concentration}")
            print(f"  元数据: {metadata}")
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            # 验证返回的数据格式
            if idx < 5:  # 只显示前5个样本的详细信息
                print(f"  返回数据类型: image={type(image)}, concentration={type(concentration)}, metadata={type(metadata)}")
            
            return image, concentration, metadata
            
        except Exception as e:
            print(f"读取图像失败: {image_path} - {e}")
            print(f"  索引: {idx}")
            print(f"  浓度值: {concentration}")
            print(f"  元数据: {metadata}")
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
        bg_types = [meta.get('bg_type', 'unknown') for meta in self.metadata]
        stats['bg_types'] = {bg: bg_types.count(bg) for bg in set(bg_types)}
        
        # 统计功率类型
        powers = [meta.get('power', 'unknown') for meta in self.metadata]
        stats['powers'] = {power: powers.count(power) for power in set(powers)}
        
        # 统计距离
        distances = [meta.get('distance', 'unknown') for meta in self.metadata]
        stats['distances'] = {dist: distances.count(dist) for dist in set(distances)}
        
        # 统计检测置信度
        confidences = [meta.get('detection_confidence', 0.0) for meta in self.metadata]
        stats['detection_confidence'] = {
            'min': min(confidences),
            'max': max(confidences),
            'mean': np.mean(confidences),
            'std': np.std(confidences)
        }
        
        return stats


def detect_feature_datasets(base_path: str = ".", version_filter: str = None) -> List[Dict]:
    """
    检测特征数据集（支持版本过滤）
    
    Args:
        base_path: 搜索基础路径
        version_filter: 版本过滤器 ('v1', 'v2', 'latest', None表示全部)
        
    Returns:
        数据集信息列表，按版本和时间排序
    """
    print(f"🔍 检测特征数据集...")
    print(f"   搜索路径: {base_path}")
    if version_filter:
        print(f"   版本过滤: {version_filter}")
    
    datasets = []
    
    # 搜索特征数据集目录
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # 检查是否为特征数据集目录
            is_feature_dataset = False
            dataset_version = 'v1'  # 默认版本
            
            # 通过目录名判断
            if item.startswith('feature_dataset'):
                is_feature_dataset = True
                
                # 提取版本信息
                if 'v2' in item.lower():
                    dataset_version = 'v2'
                elif 'v3' in item.lower():
                    dataset_version = 'v3'
                elif 'v4' in item.lower():
                    dataset_version = 'v4'
                else:
                    # 旧版本命名方式
                    dataset_version = 'v1'
            
            # 检查目录结构
            if is_feature_dataset:
                images_dir = os.path.join(item_path, 'images')
                info_dir = os.path.join(item_path, 'original_info')
                config_file = os.path.join(item_path, 'config.json')
                
                if os.path.exists(images_dir) and os.path.exists(info_dir):
                    # 获取数据集详细信息
                    dataset_info = {
                        'path': item_path,
                        'name': item,
                        'version': dataset_version,
                        'images_dir': images_dir,
                        'info_dir': info_dir,
                        'config_file': config_file if os.path.exists(config_file) else None,
                        'creation_time': None,
                        'sample_count': 0,
                        'exclude_patterns': [],
                        'yolo_model_used': None
                    }
                    
                    # 读取配置文件（如果存在）
                    if dataset_info['config_file']:
                        try:
                            with open(dataset_info['config_file'], 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                dataset_info['creation_time'] = config.get('creation_time')
                                dataset_info['exclude_patterns'] = config.get('exclude_patterns', [])
                                dataset_info['yolo_model_used'] = config.get('yolo_model_used')
                                # 从配置文件获取准确的版本信息
                                if 'version' in config:
                                    dataset_info['version'] = config['version']
                        except Exception as e:
                            print(f"⚠️ 读取配置文件失败: {config_file} - {e}")
                    
                    # 统计样本数量
                    try:
                        image_files = [f for f in os.listdir(images_dir) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        dataset_info['sample_count'] = len(image_files)
                    except Exception:
                        dataset_info['sample_count'] = 0
                    
                    # 提取时间戳（如果配置文件中没有）
                    if not dataset_info['creation_time']:
                        timestamp_match = re.search(r'(\d{8}_\d{6})', item)
                        if timestamp_match:
                            dataset_info['creation_time'] = timestamp_match.group(1)
                    
                    datasets.append(dataset_info)
    
    # 应用版本过滤
    if version_filter:
        if version_filter == 'latest':
            # 获取最新版本
            if datasets:
                # 按版本排序（v4 > v3 > v2 > v1）
                version_order = {'v4': 4, 'v3': 3, 'v2': 2, 'v1': 1}
                datasets_by_version = {}
                
                for dataset in datasets:
                    version = dataset['version']
                    if version not in datasets_by_version:
                        datasets_by_version[version] = []
                    datasets_by_version[version].append(dataset)
                
                # 找到最高版本
                max_version = max(datasets_by_version.keys(), 
                                key=lambda v: version_order.get(v, 0))
                latest_datasets = datasets_by_version[max_version]
                
                # 在同版本中选择最新的
                if latest_datasets:
                    latest_datasets.sort(key=lambda d: d.get('creation_time', ''), reverse=True)
                    datasets = [latest_datasets[0]]
                else:
                    datasets = []
        else:
            # 过滤特定版本
            datasets = [d for d in datasets if d['version'] == version_filter]
    
    # 排序：版本 > 时间
    version_order = {'v4': 4, 'v3': 3, 'v2': 2, 'v1': 1}
    datasets.sort(key=lambda d: (
        version_order.get(d['version'], 0),
        d.get('creation_time', '')
    ), reverse=True)
    
    print(f"📊 找到 {len(datasets)} 个特征数据集:")
    for i, dataset in enumerate(datasets, 1):
        exclude_info = f", 排除: {dataset['exclude_patterns']}" if dataset['exclude_patterns'] else ""
        model_info = f", 模型: {os.path.basename(dataset['yolo_model_used'])}" if dataset['yolo_model_used'] else ""
        print(f"   {i}. {dataset['name']} ({dataset['version']}) - {dataset['sample_count']} 张{exclude_info}{model_info}")
    
    return datasets


def create_feature_dataloader(feature_dataset_path: str = None,
                             batch_size: int = 32,
                             shuffle: bool = True,
                             bg_type: Optional[str] = None,
                             power_filter: Optional[str] = None,
                             concentration_range: Optional[Tuple[float, float]] = None,
                             image_size: int = 224,
                             dataset_version: str = 'latest',
                             collate_fn: Optional[Callable] = None) -> Tuple[DataLoader, FeatureImageDataset]:
    """
    创建特征数据集加载器（支持版本选择）
    
    Args:
        feature_dataset_path: 特征数据集路径，None表示自动检测
        batch_size: 批次大小
        shuffle: 是否打乱数据
        bg_type: 过滤特定背景类型 ('bg0', 'bg1')
        power_filter: 过滤特定功率 ('20mw', '100mw', '400mw')
        concentration_range: 浓度过滤范围 (min, max)
        image_size: 图像尺寸
        dataset_version: 数据集版本 ('v1', 'v2', 'v3', 'v4', 'latest')
        collate_fn: 自定义collate函数，用于定义批次数据的组织方式
        
    Returns:
        (DataLoader, Dataset) 其中DataLoader使用指定的collate_fn组织批次数据
    """
    
    # 自动检测数据集
    if feature_dataset_path is None:
        print(f"自动检测特征数据集 (版本: {dataset_version})...")
        
        datasets = detect_feature_datasets(version_filter=dataset_version)
        if not datasets:
            available_datasets = detect_feature_datasets()
            if available_datasets:
                print("可用的数据集版本:")
                for dataset in available_datasets:
                    print(f"  - {dataset['name']} ({dataset['version']})")
            raise FileNotFoundError(f"未找到版本 {dataset_version} 的特征数据集！")
        
        # 使用第一个（最新的）数据集
        feature_dataset_path = datasets[0]['path']
        dataset_info = datasets[0]
        
        print(f"✅ 选择数据集: {dataset_info['name']} ({dataset_info['version']})")
        print(f"   样本数量: {dataset_info['sample_count']}")
        if dataset_info['exclude_patterns']:
            print(f"   排除模式: {dataset_info['exclude_patterns']}")
        if dataset_info['yolo_model_used']:
            print(f"   生成模型: {os.path.basename(dataset_info['yolo_model_used'])}")
    
    # 图像变换
    if image_size != 224:
        print(f"设置图像尺寸: {image_size}x{image_size}")
    
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
        power_filter=power_filter,
        concentration_range=concentration_range
    )
    
    # 创建数据加载器
    # Windows系统使用num_workers=0避免多进程问题
    num_workers = 0 if os.name == 'nt' else 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"✅ 特征数据加载器创建完成")
    print(f"   数据集路径: {feature_dataset_path}")
    print(f"   批次大小: {batch_size}")
    print(f"   样本总数: {len(dataset)}")
    print(f"   批次数量: {len(dataloader)}")
    
    return dataloader, dataset


def main(test_dataset_path: str = None):
    """测试特征数据集加载器
    
    Args:
        test_dataset_path: 可选的测试数据集路径
    """
    print("测试特征数据集加载器")
    
    if test_dataset_path:
        # 使用指定的测试数据集
        feature_dataset_path = test_dataset_path
        print(f"使用测试数据集: {test_dataset_path}")
    else:
        # 检测特征数据集
        feature_datasets = detect_feature_datasets()
        
        if not feature_datasets:
            print("未找到特征数据集")
            print("请先运行 create_feature_dataset.py 生成特征数据集")
            return
        
        # 使用最新的特征数据集
        latest_dataset = feature_datasets[-1]
        feature_dataset_path = latest_dataset['path']
        print(f"使用数据集: {latest_dataset['name']}")
    
    try:
        # 创建DataLoader
        dataloader, dataset = create_feature_dataloader(
            feature_dataset_path=feature_dataset_path,
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
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()