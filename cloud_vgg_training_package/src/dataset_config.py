#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集路径配置文件
统一管理项目中所有数据集路径，遵循deg格式标准
"""

import os
import re

class DatasetConfig:
    """数据集路径配置类"""
    
    # 干净数据集路径（统一为deg格式）
    CLEAN_DATASET_PATHS = [
        r"D:\2025年实验照片_no_timestamp",  # Windows主路径
        r"D:\gm\2025年实验照片_no_timestamp",  # Windows备用路径
        r"/root/autodl-tmp/2025年实验照片_no_timestamp",  # Linux云服务器
        r".\2025年实验照片_no_timestamp",  # 相对路径
        r"2025年实验照片_no_timestamp"  # 当前目录
    ]
    
    # 典型数据集路径（已统一为deg格式）
    CLASSIC_DATASET_PATH = r"D:\classic_Example"
    
    # 数据格式标准 - 更灵活的模式以处理变体
    DEG_FORMAT_PATTERN = r"15deg-\d+-[0-9.]+m-\d+-bg[01]+\w*-\d+mw\s*\(\d+\)\.(jpg|json)"
    OLD_FORMAT_PATTERN = r"15[°掳聖潞∞度゜]-\d+-[0-9.]+m-\d+-bg[01]-\d+mw \(\d+\)\.(jpg|json)"
    
    # 编码修复映射
    ENCODING_FIXES = {
        '°': 'deg',      # 标准度数符号
        '掳': 'deg',     # 常见编码错误1
        '聖': 'deg',     # 常见编码错误2  
        '潞': 'deg',     # 常见编码错误3
        '∞': 'deg',      # 其他可能编码
        '度': 'deg',     # 中文度
        '゜': 'deg',     # 另一种度数符号
    }
    
    @classmethod
    def get_clean_dataset_path(cls):
        """获取干净数据集路径（统一deg格式）"""
        for path in cls.CLEAN_DATASET_PATHS:
            if os.path.exists(path):
                return path
        return None
    
    @classmethod
    def get_classic_dataset_path(cls):
        """获取典型数据集路径（统一deg格式）"""
        if os.path.exists(cls.CLASSIC_DATASET_PATH):
            return cls.CLASSIC_DATASET_PATH
        return None
    
    @classmethod
    def get_best_dataset_path(cls):
        """获取最佳可用数据集路径（优先选择干净数据集）"""
        # 优先选择干净数据集
        clean_path = cls.get_clean_dataset_path()
        if clean_path:
            return clean_path
        
        # 备选典型数据集
        classic_path = cls.get_classic_dataset_path()
        if classic_path:
            return classic_path
        
        return None
    
    @classmethod
    def normalize_filename(cls, filename: str) -> str:
        """
        标准化文件名为deg格式
        
        Args:
            filename: 原始文件名
            
        Returns:
            标准化后的文件名
        """
        normalized = filename
        
        # 应用编码修复
        for old_char, new_char in cls.ENCODING_FIXES.items():
            normalized = normalized.replace(old_char, new_char)
            
        return normalized
    
    @classmethod
    def is_deg_format(cls, filename: str) -> bool:
        """检查文件名是否符合deg格式标准"""
        return bool(re.match(cls.DEG_FORMAT_PATTERN, filename))
    
    @classmethod
    def is_old_format(cls, filename: str) -> bool:
        """检查文件名是否为旧格式（包含度数符号）"""
        return bool(re.match(cls.OLD_FORMAT_PATTERN, filename))
    
    @classmethod
    def validate_dataset_format(cls, dataset_path: str) -> dict:
        """
        验证数据集格式一致性
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            验证结果字典
        """
        if not os.path.exists(dataset_path):
            return {"error": "数据集路径不存在"}
        
        results = {
            "total_files": 0,
            "deg_format_files": 0,
            "old_format_files": 0,
            "other_files": 0,
            "deg_format_rate": 0.0,
            "needs_conversion": []
        }
        
        # 扫描所有图像文件
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    results["total_files"] += 1
                    
                    if cls.is_deg_format(file):
                        results["deg_format_files"] += 1
                    elif cls.is_old_format(file):
                        results["old_format_files"] += 1
                        results["needs_conversion"].append(os.path.join(root, file))
                    else:
                        results["other_files"] += 1
        
        # 计算deg格式比例
        if results["total_files"] > 0:
            results["deg_format_rate"] = results["deg_format_files"] / results["total_files"]
        
        return results

# 便捷函数
def get_clean_dataset_path():
    """获取干净数据集路径（向后兼容）"""
    return DatasetConfig.get_clean_dataset_path()

def get_classic_dataset_path():
    """获取典型数据集路径（向后兼容）"""
    return DatasetConfig.get_classic_dataset_path()

def get_best_dataset_path():
    """获取最佳数据集路径（向后兼容）"""
    return DatasetConfig.get_best_dataset_path()

def normalize_filename(filename: str) -> str:
    """标准化文件名（向后兼容）"""
    return DatasetConfig.normalize_filename(filename)

# 导出主要函数和类
__all__ = [
    'DatasetConfig',
    'get_clean_dataset_path', 
    'get_classic_dataset_path',
    'get_best_dataset_path',
    'normalize_filename'
]

if __name__ == "__main__":
    print("=== 数据集路径配置测试 ===")
    print(f"干净数据集路径: {get_clean_dataset_path()}")
    print(f"典型数据集路径: {get_classic_dataset_path()}")
    print(f"最佳路径（优先干净）: {get_best_dataset_path()}") 