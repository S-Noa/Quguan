#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集名称提取工具
为训练脚本提供基于数据集的输出目录命名功能
"""


import os
import re
from pathlib import Path
from datetime import datetime

def extract_dataset_name(dataset_path):
    """
    从数据集路径中提取简洁的数据集名称
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        str: 简洁的数据集名称
    """
    if not dataset_path:
        return "unknown_dataset"
    
    dataset_path = str(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    
    # 处理不同的数据集命名模式
    patterns = [
        # 特征数据集V3格式：feature_dataset_v3_20250620_180816
        (r'feature_dataset_v3_(\d{8}_\d{6})', r'v3_\1'),
        # 特征数据集V3转换格式：feature_dataset_v3_20250620_180816_converted_20250622_123456
        (r'feature_dataset_v3_\d{8}_\d{6}_converted_(\d{8}_\d{6})', r'v3_converted_\1'),
        # 特征数据集V2格式：feature_dataset_v2_20250620_150518
        (r'feature_dataset_v2_(\d{8}_\d{6})', r'v2_\1'),
        # 通用特征数据集：feature_dataset_20250604_112151
        (r'feature_dataset_(\d{8}_\d{6})', r'feature_\1'),
        # YOLO数据集：yolo_dataset_bg1_400mw_full_20250619_174407
        (r'yolo_dataset_([^_]+_[^_]+)_full_(\d{8}_\d{6})', r'yolo_\1_\2'),
        # 增强数据集：augmented_yolo_dataset_20250619_151224
        (r'augmented_yolo_dataset_(\d{8}_\d{6})', r'aug_yolo_\1'),
        # 测试数据集：test_dataset_20250620_100000
        (r'test_dataset_(\d{8}_\d{6})', r'test_\1'),
    ]
    
    # 尝试匹配已知模式
    for pattern, replacement in patterns:
        match = re.search(pattern, dataset_name)
        if match:
            return re.sub(pattern, replacement, dataset_name)
    
    # 如果没有匹配，尝试提取简短名称
    # 移除常见前缀
    clean_name = dataset_name
    prefixes_to_remove = ['feature_dataset_', 'yolo_dataset_', 'dataset_']
    for prefix in prefixes_to_remove:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break
    
    # 限制长度并移除特殊字符
    clean_name = re.sub(r'[^\w\-]', '_', clean_name)
    if len(clean_name) > 30:
        clean_name = clean_name[:30]
    
    return clean_name or "dataset"

def generate_training_output_dir(model_name, dataset_path, training_mode=None, bg_filter=None, power_filter=None):
    """
    生成基于数据集和训练模式的输出目录名
    
    Args:
        model_name: 模型名称 (baseline_cnn, enhanced_cnn, resnet50, vgg)
        dataset_path: 数据集路径
        training_mode: 训练模式 (bg0_20mw, bg1_400mw等)
        bg_filter: 背景过滤器 (bg0, bg1)
        power_filter: 功率过滤器 (20mw, 100mw, 400mw)
        
    Returns:
        str: 输出目录名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取数据集名称
    dataset_name = extract_dataset_name(dataset_path)
    
    # 构建模式后缀
    mode_parts = []
    
    if training_mode:
        # 使用显式的训练模式
        mode_parts.append(training_mode.replace('_', '-'))
    else:
        # 从过滤器构建模式
        if bg_filter:
            mode_parts.append(bg_filter)
        if power_filter:
            mode_parts.append(power_filter)
    
    mode_suffix = '_'.join(mode_parts) if mode_parts else 'all'
    
    # 构建完整目录名
    # 格式：{model_name}_{mode_suffix}_{dataset_name}_results_{timestamp}
    output_dir = f"{model_name}_{mode_suffix}_{dataset_name}_results_{timestamp}"
    
    return output_dir

def parse_training_mode_from_args(args):
    """
    从命令行参数中解析训练模式信息
    
    Args:
        args: 命令行参数对象
        
    Returns:
        tuple: (bg_filter, power_filter, training_mode)
    """
    bg_filter = None
    power_filter = None
    training_mode = None
    
    # 检查是否有显式的bg_mode参数（6档细分模式）
    if hasattr(args, 'bg_mode') and args.bg_mode:
        if '_' in args.bg_mode:
            # 细分模式：bg0_20mw, bg1_400mw等
            parts = args.bg_mode.split('_')
            if len(parts) == 2:
                bg_filter = parts[0]
                power_filter = parts[1]
                training_mode = args.bg_mode
        else:
            # 传统模式：bg0, bg1, all
            if args.bg_mode != 'all':
                bg_filter = args.bg_mode
            training_mode = args.bg_mode
    
    # 检查单独的bg_type和power_filter参数
    if hasattr(args, 'bg_type') and args.bg_type:
        bg_filter = args.bg_type
    
    if hasattr(args, 'power_filter') and args.power_filter:
        power_filter = args.power_filter
    
    return bg_filter, power_filter, training_mode

def get_dataset_info_string(dataset_path, bg_filter=None, power_filter=None):
    """
    生成数据集信息字符串，用于日志输出
    
    Args:
        dataset_path: 数据集路径
        bg_filter: 背景过滤器
        power_filter: 功率过滤器
        
    Returns:
        str: 数据集信息字符串
    """
    dataset_name = extract_dataset_name(dataset_path)
    
    info_parts = [f"数据集: {dataset_name}"]
    
    if bg_filter:
        info_parts.append(f"背景: {bg_filter}")
    
    if power_filter:
        info_parts.append(f"功率: {power_filter}")
    
    return " | ".join(info_parts)

def main():
    """测试函数"""
    test_cases = [
        "feature_dataset_v3_20250620_180816",
        "feature_dataset_v3_20250620_180816_converted_20250622_123456",
        "feature_dataset_v2_20250620_150518",
        "feature_dataset_20250604_112151",
        "yolo_dataset_bg1_400mw_full_20250619_174407",
        "augmented_yolo_dataset_20250619_151224",
    ]
    
    print("数据集名称提取测试:")
    print("=" * 60)
    
    for dataset_path in test_cases:
        dataset_name = extract_dataset_name(dataset_path)
        print(f"原始: {dataset_path}")
        print(f"提取: {dataset_name}")
        print()
    
    print("输出目录生成测试:")
    print("=" * 60)
    
    test_configs = [
        ("baseline_cnn", "feature_dataset_v3_20250620_180816", "bg0_400mw"),
        ("enhanced_cnn", "feature_dataset_v2_20250620_150518", None, "bg1", "100mw"),
        ("resnet50", "yolo_dataset_bg1_400mw_full_20250619_174407", "all"),
        ("vgg", "feature_dataset_v3_converted_20250622_123456", "bg0_20mw"),
    ]
    
    for config in test_configs:
        if len(config) == 3:
            model_name, dataset_path, training_mode = config
            output_dir = generate_training_output_dir(model_name, dataset_path, training_mode)
        else:
            model_name, dataset_path, training_mode, bg_filter, power_filter = config
            output_dir = generate_training_output_dir(model_name, dataset_path, training_mode, bg_filter, power_filter)
        
        print(f"模型: {model_name}")
        print(f"数据集: {dataset_path}")
        print(f"输出目录: {output_dir}")
        print()

if __name__ == "__main__":
    main() 