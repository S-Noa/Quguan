#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版时间戳移除脚本
专门解决中文路径问题
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def safe_imread(image_path):
    """安全读取图像，处理中文路径"""
    try:
        # 方法1: 直接读取
        image = cv2.imread(str(image_path))
        if image is not None:
            return image
        
        # 方法2: 使用字节流读取
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"读取失败: {image_path}, 错误: {e}")
        return None

def safe_imwrite(image_path, image):
    """安全保存图像，处理中文路径"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # 方法1: 直接保存
        success = cv2.imwrite(str(image_path), image)
        if success:
            return True
        
        # 方法2: 使用编码保存
        _, encoded_img = cv2.imencode('.jpg', image)
        with open(image_path, 'wb') as f:
            f.write(encoded_img.tobytes())
        return True
    except Exception as e:
        print(f"保存失败: {image_path}, 错误: {e}")
        return False

def remove_timestamp_mask(image, region=(0, 0, 0.42, 0.05)):
    """使用遮罩方法移除时间戳"""
    h, w = image.shape[:2]
    
    # 计算时间戳区域
    x = int(w * region[0])
    y = int(h * region[1])
    width = int(w * region[2])
    height = int(h * region[3])
    
    # 创建遮罩
    result = image.copy()
    if len(image.shape) == 3:
        # 彩色图像，用周围像素的平均颜色填充
        surrounding_region = image[max(0, y-10):y+height+10, max(0, x-10):x+width+10]
        if surrounding_region.size > 0:
            mean_color = np.mean(surrounding_region.reshape(-1, 3), axis=0)
            result[y:y+height, x:x+width] = mean_color
        else:
            result[y:y+height, x:x+width] = [0, 0, 0]  # 黑色
    else:
        # 灰度图像
        surrounding_region = image[max(0, y-10):y+height+10, max(0, x-10):x+width+10]
        if surrounding_region.size > 0:
            mean_value = np.mean(surrounding_region)
            result[y:y+height, x:x+width] = mean_value
        else:
            result[y:y+height, x:x+width] = 0
    
    return result

def process_dataset_simple():
    """简化处理整个数据集"""
    input_dir = Path("D:/2025年实验照片")
    output_dir = Path("D:/2025年实验照片_no_timestamp")
    
    print("=== 简化版时间戳移除工具 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"时间戳区域: 42%x5% (左上角)")
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(input_dir.rglob(ext))
    
    print(f"找到 {len(image_files)} 张图像")
    
    if len(image_files) == 0:
        print("❌ 未找到图像文件")
        return
    
    # 处理统计
    processed_count = 0
    failed_count = 0
    
    # 处理图像
    for image_path in tqdm(image_files, desc="处理图像"):
        try:
            # 读取图像
            image = safe_imread(image_path)
            if image is None:
                failed_count += 1
                continue
            
            # 移除时间戳
            processed_image = remove_timestamp_mask(image)
            
            # 构建输出路径，保持目录结构
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            
            # 保存图像
            if safe_imwrite(output_path, processed_image):
                processed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"处理失败 {image_path.name}: {e}")
            failed_count += 1
    
    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {processed_count} 张图像")
    print(f"处理失败: {failed_count} 张图像")
    print(f"输出目录: {output_dir}")
    
    # 创建几个对比样本
    create_comparison_samples(input_dir, output_dir)

def create_comparison_samples(input_dir, output_dir, num_samples=3):
    """创建对比样本"""
    print("\n创建对比样本...")
    
    # 找到一些图像文件
    image_files = list(input_dir.rglob('*.jpg'))[:num_samples]
    
    comparison_dir = output_dir / 'comparison_samples'
    comparison_dir.mkdir(exist_ok=True)
    
    for i, image_path in enumerate(image_files):
        try:
            # 读取原始图像
            original = safe_imread(image_path)
            if original is None:
                continue
            
            # 读取处理后的图像
            relative_path = image_path.relative_to(input_dir)
            processed_path = output_dir / relative_path
            processed = safe_imread(processed_path)
            if processed is None:
                continue
            
            # 创建对比图
            h, w = original.shape[:2]
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            comparison[:, :w] = original
            comparison[:, w:] = processed
            
            # 添加标签
            cv2.putText(comparison, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, 'Processed (42%x5%)', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 保存对比图
            comparison_filename = f'comparison_{i+1}_{image_path.stem}.jpg'
            comparison_path = comparison_dir / comparison_filename
            safe_imwrite(comparison_path, comparison)
            
        except Exception as e:
            print(f"创建对比样本失败: {e}")
    
    print(f"对比样本已保存到: {comparison_dir}")

if __name__ == "__main__":
    process_dataset_simple() 