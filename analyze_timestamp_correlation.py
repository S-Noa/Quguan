#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析数据集中时间戳与浓度值的相关性
检查是否存在时间戳泄露问题
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from font_utils import setup_chinese_font, get_labels

def extract_info_from_filename(filename):
    """从文件名提取信息"""
    # 文件名格式: 入射角-悬浮物浓度-相机高度-模拟流速-补光-光强_时间戳.jpg
    pattern = r'(\d+)-(\d+)-(\d+)-(\d+)-(bg[01])-(\d+)_(\d{8}_\d{6})\.jpg'
    match = re.match(pattern, filename)
    
    if match:
        angle, concentration, height, speed, bg_type, intensity, timestamp = match.groups()
        return {
            'filename': filename,
            'angle': int(angle),
            'concentration': int(concentration),
            'height': int(height),
            'speed': int(speed),
            'bg_type': bg_type,
            'intensity': int(intensity),
            'timestamp': timestamp,
            'datetime': datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
        }
    return None

def analyze_dataset_correlation(data_path):
    """分析数据集中时间戳与浓度的相关性"""
    
    # 检测数据路径
    possible_paths = [
        data_path,
        "D:/2025年实验照片",
        "D:/2025实验照片",
        "D:/gm/data",
        "D:/data", 
        "/mnt/d/data",
        "./data"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ 未找到数据集路径")
        return
    
    print(f"✓ 使用数据集路径: {dataset_path}")
    
    # 收集所有图像文件信息
    image_info = []
    total_files = 0
    valid_files = 0
    
    # 递归搜索所有子文件夹
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                total_files += 1
                info = extract_info_from_filename(filename)
                if info:
                    # 添加完整路径信息
                    info['full_path'] = os.path.join(root, filename)
                    info['subfolder'] = os.path.relpath(root, dataset_path)
                    image_info.append(info)
                    valid_files += 1
    
    print(f"总文件数: {total_files}")
    print(f"有效文件数: {valid_files}")
    print(f"解析成功率: {valid_files/total_files*100:.1f}%")
    
    if not image_info:
        print("❌ 未找到有效的图像文件")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(image_info)
    
    # 添加时间相关特征
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['second'] = df['datetime'].dt.second
    df['time_of_day'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']  # 一天中的秒数
    
    # 按时间排序，添加序号
    df = df.sort_values('datetime').reset_index(drop=True)
    df['sequence_order'] = df.index
    
    print(f"\n=== 数据集基本信息 ===")
    print(f"时间跨度: {df['datetime'].min()} 到 {df['datetime'].max()}")
    print(f"浓度范围: {df['concentration'].min()} - {df['concentration'].max()}")
    print(f"bg0样本数: {len(df[df['bg_type'] == 'bg0'])}")
    print(f"bg1样本数: {len(df[df['bg_type'] == 'bg1'])}")
    
    # 分析相关性
    print(f"\n=== 时间戳与浓度相关性分析 ===")
    
    # 整体相关性
    corr_sequence = pearsonr(df['sequence_order'], df['concentration'])
    corr_time = pearsonr(df['time_of_day'], df['concentration'])
    
    print(f"拍摄顺序与浓度的皮尔逊相关系数: {corr_sequence[0]:.4f} (p={corr_sequence[1]:.4f})")
    print(f"拍摄时间与浓度的皮尔逊相关系数: {corr_time[0]:.4f} (p={corr_time[1]:.4f})")
    
    # 分组分析
    for bg_type in ['bg0', 'bg1']:
        bg_data = df[df['bg_type'] == bg_type]
        if len(bg_data) > 1:
            corr_seq = pearsonr(bg_data['sequence_order'], bg_data['concentration'])
            corr_time = pearsonr(bg_data['time_of_day'], bg_data['concentration'])
            print(f"{bg_type} - 拍摄顺序与浓度相关性: {corr_seq[0]:.4f} (p={corr_seq[1]:.4f})")
            print(f"{bg_type} - 拍摄时间与浓度相关性: {corr_time[0]:.4f} (p={corr_time[1]:.4f})")
    
    # 设置中文字体
    font_available = setup_chinese_font()
    labels = get_labels(font_available)
    
    # 可视化分析
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 拍摄顺序 vs 浓度
    axes[0, 0].scatter(df['sequence_order'], df['concentration'], alpha=0.6, c=df['bg_type'].map({'bg0': 'blue', 'bg1': 'red'}))
    axes[0, 0].set_xlabel('拍摄顺序' if font_available else 'Sequence Order')
    axes[0, 0].set_ylabel('浓度值' if font_available else 'Concentration')
    axes[0, 0].set_title(f'拍摄顺序 vs 浓度 (r={corr_sequence[0]:.3f})' if font_available else f'Sequence vs Concentration (r={corr_sequence[0]:.3f})')
    
    # 2. 拍摄时间 vs 浓度
    axes[0, 1].scatter(df['time_of_day'], df['concentration'], alpha=0.6, c=df['bg_type'].map({'bg0': 'blue', 'bg1': 'red'}))
    axes[0, 1].set_xlabel('一天中的时间(秒)' if font_available else 'Time of Day (seconds)')
    axes[0, 1].set_ylabel('浓度值' if font_available else 'Concentration')
    axes[0, 1].set_title(f'拍摄时间 vs 浓度 (r={corr_time[0]:.3f})' if font_available else f'Time vs Concentration (r={corr_time[0]:.3f})')
    
    # 3. 浓度分布
    axes[1, 0].hist(df['concentration'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('浓度值' if font_available else 'Concentration')
    axes[1, 0].set_ylabel('频次' if font_available else 'Frequency')
    axes[1, 0].set_title('浓度分布' if font_available else 'Concentration Distribution')
    
    # 4. 时间分布
    axes[1, 1].hist(df['hour'], bins=24, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('拍摄小时' if font_available else 'Hour of Day')
    axes[1, 1].set_ylabel('频次' if font_available else 'Frequency')
    axes[1, 1].set_title('拍摄时间分布' if font_available else 'Time Distribution')
    
    plt.tight_layout()
    plt.savefig('timestamp_correlation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 相关性分析图已保存: timestamp_correlation_analysis.png")
    
    # 详细统计
    print(f"\n=== 详细统计信息 ===")
    print("浓度值统计:")
    print(df['concentration'].describe())
    
    print("\n按bg_type分组的浓度统计:")
    print(df.groupby('bg_type')['concentration'].describe())
    
    # 检查是否存在明显的时间模式
    print(f"\n=== 时间模式检查 ===")
    
    # 检查是否按浓度顺序拍摄
    concentration_changes = np.diff(df['concentration'])
    monotonic_increases = np.sum(concentration_changes > 0)
    monotonic_decreases = np.sum(concentration_changes < 0)
    no_changes = np.sum(concentration_changes == 0)
    
    print(f"浓度递增次数: {monotonic_increases}")
    print(f"浓度递减次数: {monotonic_decreases}")
    print(f"浓度不变次数: {no_changes}")
    
    # 判断是否存在泄露风险
    risk_level = "低"
    if abs(corr_sequence[0]) > 0.3 or abs(corr_time[0]) > 0.3:
        risk_level = "高"
    elif abs(corr_sequence[0]) > 0.1 or abs(corr_time[0]) > 0.1:
        risk_level = "中"
    
    print(f"\n=== 时间戳泄露风险评估 ===")
    print(f"风险等级: {risk_level}")
    
    if risk_level == "高":
        print("⚠️  警告: 检测到高风险的时间戳泄露!")
        print("建议:")
        print("1. 立即对图像进行预处理，移除时间戳")
        print("2. 重新训练模型")
        print("3. 验证新模型的Grad-CAM关注区域")
    elif risk_level == "中":
        print("⚠️  注意: 检测到中等风险的时间戳泄露")
        print("建议考虑移除时间戳并重新训练")
    else:
        print("✓ 时间戳泄露风险较低，但仍建议移除时间戳以确保模型学习正确特征")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析数据集时间戳与浓度相关性')
    parser.add_argument('--data_path', type=str, default='D:/2025年实验照片', help='数据集路径')
    
    args = parser.parse_args()
    
    print("=== 数据集时间戳泄露分析 ===")
    df = analyze_dataset_correlation(args.data_path) 