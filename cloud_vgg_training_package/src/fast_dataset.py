#!/usr/bin/env python3
"""
快速数据集加载器 - 提供多种图像验证策略
包含：缓存验证、跳过验证、采样验证、并行验证等方案
"""

import os
import re
import json
import time
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path


class FastImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, bg_type=None, 
                 validation_mode='cache', cache_file=None, 
                 sample_validation_ratio=0.1, max_workers=4):
        """
        快速图像数据集加载器
        
        参数:
            root_dir (string): 图像文件夹的根目录
            transform (callable, optional): 可选的图像转换
            bg_type (str, optional): 'bg0'或'bg1'，只加载对应补光类型的图片
            validation_mode (str): 验证模式
                - 'none': 跳过所有验证 (最快)
                - 'cache': 使用缓存验证结果 (推荐)
                - 'sample': 随机采样验证 (平衡方案)
                - 'parallel': 并行验证 (高CPU利用率)
                - 'full': 完整验证 (最安全但最慢)
            cache_file (str): 缓存文件路径
            sample_validation_ratio (float): 采样验证比例 (0.0-1.0)
            max_workers (int): 并行验证的线程数
        """
        self.root_dir = root_dir
        self.transform = transform
        self.bg_type = bg_type
        self.validation_mode = validation_mode
        self.sample_validation_ratio = sample_validation_ratio
        self.max_workers = max_workers
        
        # 设置缓存文件路径
        if cache_file is None:
            root_hash = hashlib.md5(root_dir.encode()).hexdigest()[:8]
            bg_suffix = f"_{bg_type}" if bg_type else "_all"
            cache_file = f"dataset_cache_{root_hash}{bg_suffix}.json"
        self.cache_file = cache_file
        
        print(f"🚀 初始化快速数据集 (模式: {validation_mode})")
        print(f"📁 数据路径: {root_dir}")
        print(f"🏷️ 数据类型: {bg_type or 'all'}")
        
        start_time = time.time()
        
        # 根据验证模式选择不同的初始化策略
        if validation_mode == 'none':
            self._init_no_validation()
        elif validation_mode == 'cache':
            self._init_with_cache()
        elif validation_mode == 'sample':
            self._init_sample_validation()
        elif validation_mode == 'parallel':
            self._init_parallel_validation()
        elif validation_mode == 'full':
            self._init_full_validation()
        else:
            raise ValueError(f"未知的验证模式: {validation_mode}")
        
        init_time = time.time() - start_time
        print(f"⏱️ 数据集初始化完成，耗时: {init_time:.2f}s")
        
        if self.concentrations:
            self._print_statistics()
        else:
            print(f"❌ 未找到任何有效的图像文件!")
    
    def _find_candidate_files(self):
        """查找候选图像文件"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        concentration_pattern = r'\d+°-(\d+)-'
        
        candidate_files = []
        candidate_concentrations = []
        
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    if self.bg_type is not None and self.bg_type not in file:
                        continue
                    
                    file_path = os.path.join(root, file)
                    match = re.search(concentration_pattern, file)
                    if match:
                        concentration = float(match.group(1))
                        candidate_files.append(file_path)
                        candidate_concentrations.append(concentration)
        
        print(f"📋 找到 {len(candidate_files)} 个候选文件")
        return candidate_files, candidate_concentrations
    
    def _init_no_validation(self):
        """跳过验证模式 - 最快但有风险"""
        print("⚡ 跳过图像验证 (风险模式)")
        
        candidate_files, candidate_concentrations = self._find_candidate_files()
        self.image_files = candidate_files
        self.concentrations = candidate_concentrations
        
        print("⚠️ 警告: 未验证图像完整性，可能在训练时遇到损坏文件")
    
    def _init_with_cache(self):
        """缓存验证模式 - 推荐方案"""
        print("💾 使用缓存验证模式")
        
        candidate_files, candidate_concentrations = self._find_candidate_files()
        
        # 尝试加载缓存
        cache_data = self._load_cache()
        if cache_data:
            print(f"📥 加载缓存文件: {self.cache_file}")
            cached_valid_files = set(cache_data.get('valid_files', []))
            cached_invalid_files = set(cache_data.get('invalid_files', []))
            
            # 筛选有效文件
            self.image_files = []
            self.concentrations = []
            need_validation = []
            need_validation_conc = []
            
            for file_path, conc in zip(candidate_files, candidate_concentrations):
                if file_path in cached_valid_files:
                    # 缓存中标记为有效
                    if os.path.exists(file_path):  # 快速检查文件是否存在
                        self.image_files.append(file_path)
                        self.concentrations.append(conc)
                elif file_path in cached_invalid_files:
                    # 缓存中标记为无效，跳过
                    continue
                else:
                    # 新文件，需要验证
                    need_validation.append(file_path)
                    need_validation_conc.append(conc)
            
            print(f"✅ 从缓存中获得 {len(self.image_files)} 个已验证文件")
            
            # 验证新文件
            if need_validation:
                print(f"🔍 验证 {len(need_validation)} 个新文件...")
                valid_new, invalid_new = self._validate_files(need_validation, need_validation_conc)
                self.image_files.extend([f for f, _ in valid_new])
                self.concentrations.extend([c for _, c in valid_new])
                
                # 更新缓存
                self._update_cache(valid_new, invalid_new)
        else:
            print("📝 缓存文件不存在，进行完整验证...")
            valid_files, invalid_files = self._validate_files(candidate_files, candidate_concentrations)
            self.image_files = [f for f, _ in valid_files]
            self.concentrations = [c for _, c in valid_files]
            
            # 创建缓存
            self._create_cache(valid_files, invalid_files)
    
    def _init_sample_validation(self):
        """采样验证模式 - 平衡方案"""
        print(f"🎯 采样验证模式 (验证比例: {self.sample_validation_ratio*100:.1f}%)")
        
        candidate_files, candidate_concentrations = self._find_candidate_files()
        
        if self.sample_validation_ratio >= 1.0:
            # 完整验证
            valid_files, _ = self._validate_files(candidate_files, candidate_concentrations)
            self.image_files = [f for f, _ in valid_files]
            self.concentrations = [c for _, c in valid_files]
        else:
            # 采样验证
            import random
            sample_size = max(1, int(len(candidate_files) * self.sample_validation_ratio))
            sample_indices = random.sample(range(len(candidate_files)), sample_size)
            
            sample_files = [candidate_files[i] for i in sample_indices]
            sample_concs = [candidate_concentrations[i] for i in sample_indices]
            
            print(f"🔍 验证 {len(sample_files)} 个采样文件...")
            valid_samples, invalid_samples = self._validate_files(sample_files, sample_concs)
            
            # 计算损坏率
            corruption_rate = len(invalid_samples) / len(sample_files) if sample_files else 0
            print(f"📊 采样损坏率: {corruption_rate*100:.2f}%")
            
            if corruption_rate < 0.05:  # 损坏率低于5%
                print("✅ 损坏率较低，跳过其余验证")
                self.image_files = candidate_files
                self.concentrations = candidate_concentrations
            else:
                print(f"⚠️ 损坏率较高 ({corruption_rate*100:.2f}%)，进行完整验证...")
                valid_files, _ = self._validate_files(candidate_files, candidate_concentrations)
                self.image_files = [f for f, _ in valid_files]
                self.concentrations = [c for _, c in valid_files]
    
    def _init_parallel_validation(self):
        """并行验证模式 - 高CPU利用率"""
        print(f"🔄 并行验证模式 (线程数: {self.max_workers})")
        
        candidate_files, candidate_concentrations = self._find_candidate_files()
        valid_files, _ = self._validate_files_parallel(candidate_files, candidate_concentrations)
        
        self.image_files = [f for f, _ in valid_files]
        self.concentrations = [c for _, c in valid_files]
    
    def _init_full_validation(self):
        """完整验证模式 - 最安全但最慢"""
        print("🛡️ 完整验证模式")
        
        candidate_files, candidate_concentrations = self._find_candidate_files()
        valid_files, invalid_files = self._validate_files(candidate_files, candidate_concentrations)
        
        self.image_files = [f for f, _ in valid_files]
        self.concentrations = [c for _, c in valid_files]
        
        print(f"🗑️ 损坏文件数: {len(invalid_files)}")
    
    def _validate_single_file(self, file_path):
        """验证单个文件"""
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except (OSError, IOError, Image.UnidentifiedImageError):
            return False
    
    def _validate_files(self, files, concentrations):
        """验证文件列表"""
        valid_files = []
        invalid_files = []
        
        for file_path, conc in zip(files, concentrations):
            if self._validate_single_file(file_path):
                valid_files.append((file_path, conc))
            else:
                invalid_files.append((file_path, conc))
        
        return valid_files, invalid_files
    
    def _validate_files_parallel(self, files, concentrations):
        """并行验证文件列表"""
        valid_files = []
        invalid_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交验证任务
            future_to_file = {
                executor.submit(self._validate_single_file, file_path): (file_path, conc)
                for file_path, conc in zip(files, concentrations)
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path, conc = future_to_file[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        valid_files.append((file_path, conc))
                    else:
                        invalid_files.append((file_path, conc))
                except Exception as e:
                    print(f"验证错误 {file_path}: {e}")
                    invalid_files.append((file_path, conc))
        
        return valid_files, invalid_files
    
    def _load_cache(self):
        """加载缓存文件"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ 缓存加载失败: {e}")
        return None
    
    def _create_cache(self, valid_files, invalid_files):
        """创建缓存文件"""
        cache_data = {
            'valid_files': [f for f, _ in valid_files],
            'invalid_files': [f for f, _ in invalid_files],
            'creation_time': time.time(),
            'root_dir': self.root_dir,
            'bg_type': self.bg_type
        }
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"💾 缓存已保存: {self.cache_file}")
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")
    
    def _update_cache(self, new_valid_files, new_invalid_files):
        """更新缓存文件"""
        cache_data = self._load_cache() or {'valid_files': [], 'invalid_files': []}
        
        cache_data['valid_files'].extend([f for f, _ in new_valid_files])
        cache_data['invalid_files'].extend([f for f, _ in new_invalid_files])
        cache_data['update_time'] = time.time()
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"💾 缓存已更新")
        except Exception as e:
            print(f"⚠️ 缓存更新失败: {e}")
    
    def _print_statistics(self):
        """打印统计信息"""
        min_conc = min(self.concentrations)
        max_conc = max(self.concentrations)
        avg_conc = sum(self.concentrations) / len(self.concentrations)
        
        print(f"📊 数据集统计:")
        print(f"   有效图像: {len(self.image_files)} 张")
        print(f"   浓度范围: {min_conc} - {max_conc}")
        print(f"   平均浓度: {avg_conc:.2f}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        concentration = self.concentrations[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"运行时图像读取错误: {img_path} - {e}")
            # 创建黑色图像作为应急措施
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, concentration


# 为了兼容性，创建一个别名
class OptimizedImageDataset(FastImageDataset):
    """优化图像数据集 - FastImageDataset的别名"""
    pass


def get_recommended_dataset(root_dir, transform=None, bg_type=None):
    """
    获取推荐的数据集配置
    
    根据数据集大小自动选择最优的验证策略
    """
    # 快速统计文件数量
    file_count = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if bg_type is None or bg_type in file:
                    file_count += 1
    
    print(f"📊 预估文件数: {file_count}")
    
    # 根据文件数量选择策略
    if file_count < 1000:
        print("💡 推荐: 完整验证模式 (文件数较少)")
        return FastImageDataset(root_dir, transform, bg_type, validation_mode='full')
    elif file_count < 10000:
        print("💡 推荐: 缓存验证模式 (平衡性能)")
        return FastImageDataset(root_dir, transform, bg_type, validation_mode='cache')
    else:
        print("💡 推荐: 采样验证模式 (大数据集)")
        return FastImageDataset(root_dir, transform, bg_type, validation_mode='sample', 
                               sample_validation_ratio=0.05) 