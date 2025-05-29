#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集路径配置文件
统一管理项目中所有数据集路径
"""

import os

class DatasetConfig:
    """数据集路径配置类"""
    
    # 干净数据集路径（移除时间戳后）
    CLEAN_DATASET_PATHS = [
        r"/root/autodl-tmp/2025年实验照片_no_timestamp",  # Linux云服务器
        r"D:\gm\2025年实验照片_no_timestamp",  # Windows本地
        r"D:\2025年实验照片_no_timestamp",  # Windows备用
        r".\2025年实验照片_no_timestamp",  # 相对路径
        r"2025年实验照片_no_timestamp"  # 当前目录
    ]
    
    # 原始数据集路径（包含时间戳）
    ORIGINAL_DATASET_PATHS = [
        r"/root/autodl-tmp/2025年实验照片",  # Linux云服务器
        r"D:\gm\2025年实验照片",  # Windows本地
        r"D:\2025年实验照片",  # Windows备用
        r".\2025年实验照片",  # 相对路径
        r"2025年实验照片"  # 当前目录
    ]
    
    @classmethod
    def get_clean_dataset_path(cls):
        """获取干净数据集路径（优先选择）"""
        for path in cls.CLEAN_DATASET_PATHS:
            if os.path.exists(path):
                return path
        return None
    
    @classmethod
    def get_original_dataset_path(cls):
        """获取原始数据集路径"""
        for path in cls.ORIGINAL_DATASET_PATHS:
            if os.path.exists(path):
                return path
        return None
    
    @classmethod
    def get_best_dataset_path(cls, prefer_clean=True):
        """
        获取最佳数据集路径
        
        Args:
            prefer_clean (bool): 是否优先选择干净数据集
            
        Returns:
            str: 数据集路径，如果都不存在则返回None
        """
        if prefer_clean:
            # 优先尝试干净数据集
            clean_path = cls.get_clean_dataset_path()
            if clean_path:
                print(f"✓ 使用干净数据集: {clean_path}")
                return clean_path
            
            # 备用原始数据集
            original_path = cls.get_original_dataset_path()
            if original_path:
                print(f"⚠️ 干净数据集不存在，使用原始数据集: {original_path}")
                print("  建议先运行时间戳移除工具生成干净数据集")
                return original_path
        else:
            # 优先使用原始数据集（如时间戳分析工具）
            original_path = cls.get_original_dataset_path()
            if original_path:
                print(f"✓ 使用原始数据集: {original_path}")
                return original_path
            
            # 备用干净数据集
            clean_path = cls.get_clean_dataset_path()
            if clean_path:
                print(f"⚠️ 原始数据集不存在，使用干净数据集: {clean_path}")
                return clean_path
        
        print("❌ 未找到任何数据集！")
        print("可能的路径:")
        all_paths = cls.CLEAN_DATASET_PATHS + cls.ORIGINAL_DATASET_PATHS
        for path in all_paths:
            print(f"  {path}")
        return None
    
    @classmethod
    def get_output_path(cls, suffix="_no_timestamp"):
        """
        获取输出路径（基于原始数据集路径）
        
        Args:
            suffix (str): 输出目录后缀
            
        Returns:
            str: 输出路径
        """
        original_path = cls.get_original_dataset_path()
        if original_path:
            return original_path + suffix
        
        # 如果原始路径不存在，使用默认路径
        return f"D:/2025年实验照片{suffix}"

# 便捷函数
def get_dataset_path(prefer_clean=True):
    """获取数据集路径的便捷函数"""
    return DatasetConfig.get_best_dataset_path(prefer_clean=prefer_clean)

def get_clean_dataset_path():
    """获取干净数据集路径的便捷函数"""
    return DatasetConfig.get_clean_dataset_path()

def get_original_dataset_path():
    """获取原始数据集路径的便捷函数"""
    return DatasetConfig.get_original_dataset_path()

if __name__ == "__main__":
    print("=== 数据集路径配置测试 ===")
    print(f"干净数据集路径: {get_clean_dataset_path()}")
    print(f"原始数据集路径: {get_original_dataset_path()}")
    print(f"最佳路径（优先干净）: {get_dataset_path(prefer_clean=True)}")
    print(f"最佳路径（优先原始）: {get_dataset_path(prefer_clean=False)}") 