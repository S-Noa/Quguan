#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像时间戳移除预处理脚本
解决模型关注时间戳而非光斑的数据泄露问题
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class TimestampRemover:
    def __init__(self, input_dir, output_dir, timestamp_region=None, method='crop'):
        """
        初始化时间戳移除器
        
        Args:
            input_dir: 输入图像目录
            output_dir: 输出图像目录
            timestamp_region: 时间戳区域 (x, y, width, height)，如果为None则自动检测
            method: 移除方法 ('crop', 'mask', 'inpaint')
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.method = method
        self.processed_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
        
        # 默认时间戳区域（左上角）
        if timestamp_region is None:
            # 根据时间戳特点调整：宽度较小，长度较大（水平排列）
            # 时间戳通常在左上角，呈水平条状分布
            self.timestamp_region = (0, 0, 0.42, 0.05)  # (x_ratio, y_ratio, width_ratio, height_ratio)
        else:
            self.timestamp_region = timestamp_region
    
    def detect_timestamp_region(self, sample_dir, sample_size=10):
        """
        自动检测时间戳区域（通过分析多张图像的差异）
        """
        print("正在自动检测时间戳区域...")
        
        # 读取样本图像
        image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sample_files = image_files[:min(sample_size, len(image_files))]
        
        images = []
        for file in sample_files:
            img_path = os.path.join(sample_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        
        if len(images) < 2:
            print("样本图像不足，使用默认时间戳区域")
            return self.timestamp_region
        
        # 计算图像差异，找出变化最大的区域（通常是时间戳）
        h, w = images[0].shape
        diff_map = np.zeros((h, w), dtype=np.float32)
        
        for i in range(len(images)-1):
            diff = cv2.absdiff(images[i], images[i+1]).astype(np.float32)
            diff_map += diff
        
        diff_map /= (len(images) - 1)
        
        # 找出差异最大的区域
        threshold = np.percentile(diff_map, 95)
        binary_diff = (diff_map > threshold).astype(np.uint8)
        
        # 找轮廓
        contours, _ = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找最大的轮廓（假设是时间戳）
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            
            # 转换为比例
            detected_region = (x/w, y/h, w_box/w, h_box/h)
            print(f"检测到时间戳区域: {detected_region}")
            return detected_region
        
        print("未能自动检测时间戳区域，使用默认区域")
        return self.timestamp_region
    
    def remove_timestamp_crop(self, image):
        """通过裁剪移除时间戳"""
        h, w = image.shape[:2]
        
        # 计算裁剪区域
        x_start = int(w * self.timestamp_region[2])  # 跳过时间戳宽度
        y_start = int(h * self.timestamp_region[3])  # 跳过时间戳高度
        
        # 裁剪图像
        cropped = image[y_start:, x_start:]
        
        # 调整大小回原始尺寸
        resized = cv2.resize(cropped, (w, h))
        return resized
    
    def remove_timestamp_mask(self, image):
        """通过遮罩移除时间戳"""
        h, w = image.shape[:2]
        
        # 计算时间戳区域
        x = int(w * self.timestamp_region[0])
        y = int(h * self.timestamp_region[1])
        width = int(w * self.timestamp_region[2])
        height = int(h * self.timestamp_region[3])
        
        # 创建遮罩（用黑色或平均颜色填充）
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
    
    def remove_timestamp_inpaint(self, image):
        """通过图像修复移除时间戳"""
        h, w = image.shape[:2]
        
        # 计算时间戳区域
        x = int(w * self.timestamp_region[0])
        y = int(h * self.timestamp_region[1])
        width = int(w * self.timestamp_region[2])
        height = int(h * self.timestamp_region[3])
        
        # 创建遮罩
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y:y+height, x:x+width] = 255
        
        # 使用OpenCV的图像修复
        if len(image.shape) == 3:
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        else:
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def process_single_image(self, input_path, output_path):
        """处理单张图像"""
        try:
            # 读取图像 - 使用更健壮的方法
            image = None
            
            # 方法1: 直接读取
            image = cv2.imread(input_path)
            
            # 方法2: 如果失败，尝试使用字节方式读取
            if image is None:
                try:
                    with open(input_path, 'rb') as f:
                        image_data = f.read()
                    image_array = np.frombuffer(image_data, dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                except Exception as e:
                    raise ValueError(f"无法读取图像 (方法2): {input_path}, 错误: {e}")
            
            if image is None:
                raise ValueError(f"无法读取图像: {input_path}")
            
            # 根据方法移除时间戳
            if self.method == 'crop':
                processed = self.remove_timestamp_crop(image)
            elif self.method == 'mask':
                processed = self.remove_timestamp_mask(image)
            elif self.method == 'inpaint':
                processed = self.remove_timestamp_inpaint(image)
            else:
                raise ValueError(f"未知的处理方法: {self.method}")
            
            # 保存处理后的图像
            success = cv2.imwrite(output_path, processed)
            if not success:
                raise ValueError(f"无法保存图像: {output_path}")
            
            with self.lock:
                self.processed_count += 1
            
            return True
            
        except Exception as e:
            print(f"处理图像失败 {os.path.basename(input_path)}: {e}")
            with self.lock:
                self.failed_count += 1
            return False
    
    def process_dataset(self, max_workers=4, auto_detect=True):
        """处理整个数据集"""
        
        # 检查输入目录
        if not os.path.exists(self.input_dir):
            raise ValueError(f"输入目录不存在: {self.input_dir}")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 递归获取所有图像文件
        image_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # 保存相对路径信息
                    rel_path = os.path.relpath(root, self.input_dir)
                    if rel_path == '.':
                        rel_path = ''
                    image_files.append((file, rel_path))
        
        if not image_files:
            raise ValueError(f"输入目录中没有找到图像文件: {self.input_dir}")
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 自动检测时间戳区域（使用第一个子文件夹的样本）
        if auto_detect:
            sample_dir = self.input_dir
            # 如果有子文件夹，使用第一个子文件夹进行检测
            subdirs = [d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))]
            if subdirs:
                sample_dir = os.path.join(self.input_dir, subdirs[0])
            self.timestamp_region = self.detect_timestamp_region(sample_dir)
        
        print(f"使用时间戳区域: {self.timestamp_region}")
        print(f"使用处理方法: {self.method}")
        print(f"开始处理...")
        
        # 多线程处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for file, rel_path in image_files:
                input_path = os.path.join(self.input_dir, rel_path, file)
                output_subdir = os.path.join(self.output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, file)
                future = executor.submit(self.process_single_image, input_path, output_path)
                futures.append(future)
            
            # 显示进度
            with tqdm(total=len(futures), desc="处理图像") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)
        
        print(f"\n处理完成!")
        print(f"成功处理: {self.processed_count} 张图像")
        print(f"处理失败: {self.failed_count} 张图像")
        print(f"输出目录: {self.output_dir}")
        print(f"目录结构已保持一致")
    
    def create_comparison_samples(self, num_samples=5):
        """创建处理前后的对比样本"""
        print("创建对比样本...")
        
        # 递归获取所有图像文件
        all_image_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(root, self.input_dir)
                    if rel_path == '.':
                        rel_path = ''
                    all_image_files.append((file, rel_path))
        
        sample_files = all_image_files[:min(num_samples, len(all_image_files))]
        
        comparison_dir = os.path.join(self.output_dir, 'comparison_samples')
        os.makedirs(comparison_dir, exist_ok=True)
        
        for i, (file, rel_path) in enumerate(sample_files):
            input_path = os.path.join(self.input_dir, rel_path, file)
            original = cv2.imread(input_path)
            
            if original is not None:
                # 处理图像
                if self.method == 'crop':
                    processed = self.remove_timestamp_crop(original)
                elif self.method == 'mask':
                    processed = self.remove_timestamp_mask(original)
                elif self.method == 'inpaint':
                    processed = self.remove_timestamp_inpaint(original)
                
                # 创建对比图
                h, w = original.shape[:2]
                comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
                comparison[:, :w] = original
                comparison[:, w:] = processed
                
                # 添加标签
                cv2.putText(comparison, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(comparison, 'Processed', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 保存对比图，包含子文件夹信息
                safe_rel_path = rel_path.replace('/', '_').replace('\\', '_') if rel_path else ''
                comparison_filename = f'comparison_{i+1}_{safe_rel_path}_{file}' if safe_rel_path else f'comparison_{i+1}_{file}'
                comparison_path = os.path.join(comparison_dir, comparison_filename)
                cv2.imwrite(comparison_path, comparison)
        
        print(f"对比样本已保存到: {comparison_dir}")

def main():
    parser = argparse.ArgumentParser(description='移除图像中的时间戳')
    parser.add_argument('--input_dir', type=str, default='D:/2025年实验照片', help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='D:/2025年实验照片_no_timestamp', help='输出图像目录')
    parser.add_argument('--method', type=str, choices=['crop', 'mask', 'inpaint'], default='mask', 
                       help='时间戳移除方法')
    parser.add_argument('--workers', type=int, default=4, help='并行处理线程数')
    parser.add_argument('--no_auto_detect', action='store_true', help='不自动检测时间戳区域')
    parser.add_argument('--create_samples', action='store_true', help='创建处理前后对比样本')
    
    args = parser.parse_args()
    
    print("=== 图像时间戳移除工具 ===")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理方法: {args.method}")
    
    # 创建时间戳移除器
    remover = TimestampRemover(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        method=args.method
    )
    
    try:
        # 处理数据集
        remover.process_dataset(
            max_workers=args.workers,
            auto_detect=not args.no_auto_detect
        )
        
        # 创建对比样本
        if args.create_samples:
            remover.create_comparison_samples()
        
        print("\n=== 下一步建议 ===")
        print("1. 检查输出目录中的处理结果")
        print("2. 使用处理后的数据集重新训练模型:")
        print(f"   python train.py --data_path {args.output_dir} --group allgroup")
        print("3. 训练完成后生成新的Grad-CAM可视化验证关注区域")
        
    except Exception as e:
        print(f"处理失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 