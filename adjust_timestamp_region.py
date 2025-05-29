#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式时间戳区域调整工具
帮助用户精确设定时间戳移除区域
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from font_utils import setup_chinese_font, get_labels

class TimestampRegionAdjuster:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.timestamp_region = (0, 0, 0.25, 0.12)  # 默认区域
        
    def load_image(self):
        """加载图像"""
        # 使用健壮的方法读取图像
        self.image = cv2.imread(self.image_path)
        
        if self.image is None:
            try:
                with open(self.image_path, 'rb') as f:
                    image_data = f.read()
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                self.image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except Exception as e:
                raise ValueError(f"无法读取图像: {e}")
        
        if self.image is None:
            raise ValueError(f"无法读取图像: {self.image_path}")
        
        # 转换为RGB
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print(f"✓ 成功加载图像，尺寸: {self.image.shape}")
    
    def test_different_regions(self):
        """测试不同大小的时间戳区域"""
        if self.image is None:
            self.load_image()
        
        # 设置字体
        font_available = setup_chinese_font()
        
        # 测试不同的区域大小 - 针对水平条状时间戳优化
        test_regions = [
            (0, 0, 0.25, 0.06),  # 小区域（窄高度）
            (0, 0, 0.35, 0.08),  # 中等区域（标准高度）
            (0, 0, 0.40, 0.08),  # 大区域（当前默认）
            (0, 0, 0.45, 0.10),  # 更大区域
            (0, 0, 0.50, 0.12),  # 最大区域
        ]
        
        region_names = [
            '小区域 (25%x6%) - 窄条',
            '中等区域 (35%x8%) - 标准条', 
            '大区域 (40%x8%) - 当前默认',
            '更大区域 (45%x10%) - 宽条',
            '最大区域 (50%x12%) - 最宽条'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('时间戳区域大小测试' if font_available else 'Timestamp Region Size Test', fontsize=16)
        
        # 显示原始图像
        axes[0, 0].imshow(self.image)
        axes[0, 0].set_title('原始图像' if font_available else 'Original Image')
        axes[0, 0].axis('off')
        
        h, w = self.image.shape[:2]
        
        # 测试不同区域
        for i, (region, name) in enumerate(zip(test_regions, region_names)):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            # 创建遮罩预览
            preview = self.image.copy()
            x = int(w * region[0])
            y = int(h * region[1])
            width = int(w * region[2])
            height = int(h * region[3])
            
            # 添加半透明红色覆盖层显示将被移除的区域
            overlay = preview.copy()
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 0, 0), -1)
            preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)
            
            axes[row, col].imshow(preview)
            axes[row, col].set_title(name)
            axes[row, col].axis('off')
            
            # 添加区域边框
            rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
            axes[row, col].add_patch(rect)
        
        plt.tight_layout()
        plt.savefig('timestamp_region_test.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n=== 区域测试完成 ===")
        print("红色区域表示将被移除的时间戳区域")
        print("请查看 timestamp_region_test.png 选择最适合的区域大小")
        
        return test_regions, region_names
    
    def create_custom_region_preview(self, x_ratio, y_ratio, width_ratio, height_ratio):
        """创建自定义区域预览"""
        if self.image is None:
            self.load_image()
        
        font_available = setup_chinese_font()
        
        h, w = self.image.shape[:2]
        x = int(w * x_ratio)
        y = int(h * y_ratio)
        width = int(w * width_ratio)
        height = int(h * height_ratio)
        
        # 创建预览图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始图像
        axes[0].imshow(self.image)
        axes[0].set_title('原始图像' if font_available else 'Original Image')
        axes[0].axis('off')
        
        # 带区域标记的图像
        preview = self.image.copy()
        overlay = preview.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 0, 0), -1)
        preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)
        
        axes[1].imshow(preview)
        axes[1].set_title(f'时间戳区域预览 ({width_ratio:.1%}x{height_ratio:.1%})' if font_available else f'Timestamp Region Preview ({width_ratio:.1%}x{height_ratio:.1%})')
        axes[1].axis('off')
        
        # 添加区域边框
        rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        
        plt.tight_layout()
        plt.savefig('custom_timestamp_region.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 自定义区域预览已保存: custom_timestamp_region.png")
        print(f"区域参数: x={x_ratio:.3f}, y={y_ratio:.3f}, width={width_ratio:.3f}, height={height_ratio:.3f}")
        
        return (x_ratio, y_ratio, width_ratio, height_ratio)

def main():
    # 数据路径 - 优先使用原始数据集（因为这是时间戳调整工具）
    possible_input_dirs = [
        "D:/2025年实验照片",  # 原始数据集（用于调整时间戳区域）
        "D:/2025年实验照片_no_timestamp"  # 如果需要在干净数据集上测试
    ]
    
    input_dir = None
    for dir_path in possible_input_dirs:
        if os.path.exists(dir_path):
            input_dir = dir_path
            print(f"使用输入目录: {input_dir}")
            break
    
    if input_dir is None:
        print("错误：未找到数据集目录！")
        return
    
    # 找到第一张图像
    test_image = None
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
    
    if not test_image:
        print("❌ 未找到测试图像")
        return
    
    print(f"✓ 使用测试图像: {os.path.relpath(test_image, input_dir)}")
    
    # 创建调整器
    adjuster = TimestampRegionAdjuster(test_image)
    
    try:
        # 测试不同区域大小
        regions, names = adjuster.test_different_regions()
        
        print("\n=== 推荐设置 ===")
        print("根据您的反馈，推荐使用以下区域设置:")
        for i, (region, name) in enumerate(zip(regions, names)):
            print(f"{i+1}. {name}: {region}")
        
        print("\n=== 使用方法 ===")
        print("选择合适的区域后，可以这样运行预处理:")
        print("python remove_timestamp_preprocessing.py --method mask --create_samples")
        
        # 测试自定义区域（覆盖完整时间戳）
        print("\n=== 自定义区域测试 ===")
        print("测试水平条状区域以确保覆盖完整时间戳...")
        custom_region = adjuster.create_custom_region_preview(0, 0, 0.45, 0.10)
        print("如果这个区域合适，可以在代码中使用这些参数")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    main() 