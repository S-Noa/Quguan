#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import transforms
from cnn_model import CNNFeatureExtractor
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import os
import glob
from PIL import Image

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib的中文字体"""
    try:
        # 尝试常见的中文字体
        chinese_fonts = [
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',  # 宋体
            'KaiTi',   # 楷体
            'FangSong'  # 仿宋
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用中文字体: {font}")
                return font
        
        # 如果没有找到中文字体，使用英文
        print("❌ 未找到中文字体，将使用英文标签")
        return None
        
    except Exception as e:
        print(f"字体设置失败: {e}")
        return None

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        handle1 = self.target_layer.register_forward_hook(forward_hook)
        handle2 = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle1, handle2])
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def __call__(self, input_tensor):
        self.model.eval()
        output, _ = self.model(input_tensor)
        
        # 计算损失并反向传播
        loss = output.sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        # 计算权重和CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        
        self.remove_hooks()
        return cam

def load_model_safe(model_path, device):
    """安全加载模型"""
    print(f"Loading model: {model_path}")
    
    model = CNNFeatureExtractor().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    
    # 过滤匹配的权重
    filtered_dict = {k: v for k, v in state_dict.items() 
                    if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    print(f"✓ Model loaded successfully ({len(filtered_dict)}/{len(state_dict)} layers)")
    return model

def parse_filename(filename):
    """从文件名解析浓度值"""
    try:
        # 文件名格式: 入射角-悬浮物浓度-相机高度-模拟流速-补光-光强 (序号).jpg
        basename = os.path.basename(filename)
        # 移除 (序号).jpg 部分
        name_part = basename.split(' (')[0]
        parts = name_part.split('-')
        if len(parts) >= 6:
            concentration = float(parts[1])
            bg_type = parts[4]  # bg0 或 bg1
            return concentration, bg_type
    except Exception as e:
        print(f"Failed to parse filename {filename}: {e}")
    return None, None

def find_real_experiment_images():
    """查找真实实验图像"""
    base_path = r"D:\2025年实验照片"
    
    # 定义要查找的特定图像
    target_images = [
        # bg0类型图像 (不补光)
        {
            'pattern': '**/15°-1000-*-bg0-*.jpg',
            'description': 'bg0-concentration-1000',
            'expected_concentration': 1000,
            'bg_type': 'bg0'
        },
        {
            'pattern': '**/15°-960-*-bg0-*.jpg', 
            'description': 'bg0-concentration-960',
            'expected_concentration': 960,
            'bg_type': 'bg0'
        },
        # bg1类型图像 (补光)
        {
            'pattern': '**/15°-1000-*-bg1-*.jpg',
            'description': 'bg1-concentration-1000', 
            'expected_concentration': 1000,
            'bg_type': 'bg1'
        },
        {
            'pattern': '**/15°-960-*-bg1-*.jpg',
            'description': 'bg1-concentration-960',
            'expected_concentration': 960,
            'bg_type': 'bg1'
        }
    ]
    
    found_images = []
    
    for target in target_images:
        pattern = os.path.join(base_path, target['pattern'])
        files = glob.glob(pattern, recursive=True)
        
        if files:
            # 选择第一个找到的文件
            selected_file = files[0]
            concentration, bg_type = parse_filename(selected_file)
            
            if concentration is not None and bg_type == target['bg_type']:
                found_images.append({
                    'path': selected_file,
                    'concentration': concentration,
                    'bg_type': bg_type,
                    'description': target['description']
                })
                print(f"✓ Found: {target['description']} - {os.path.basename(selected_file)}")
            else:
                print(f"❌ Parse failed: {selected_file}")
        else:
            print(f"❌ Not found: {target['description']}")
    
    return found_images

def load_and_preprocess_image(image_path, transform):
    """加载并预处理图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image)
        return tensor, True
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None, False

def create_gradcam_visualization(model, model_name, image_info, transform, device, use_chinese=True):
    """为单个模型和图像创建Grad-CAM可视化"""
    image_path = image_info['path']
    concentration = image_info['concentration']
    bg_type = image_info['bg_type']
    description = image_info['description']
    
    print(f"  Processing image: {description}")
    
    # 加载图像
    img_tensor, success = load_and_preprocess_image(image_path, transform)
    if not success:
        return None
    
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 获取预测
    with torch.no_grad():
        prediction, _ = model(input_tensor)
        pred_value = prediction.item()
    
    print(f"    True concentration: {concentration:.1f}, Predicted: {pred_value:.1f}")
    
    # 生成Grad-CAM
    grad_cam = GradCAM(model, model.conv5)
    cam = grad_cam(input_tensor)
    
    # 处理原始图像用于显示
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    
    # 调整CAM尺寸并生成热力图
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cm.jet(cam_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 创建叠加图像
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    return {
        'img_np': img_np,
        'cam_resized': cam_resized,
        'heatmap': heatmap,
        'overlay': overlay,
        'pred_value': pred_value,
        'concentration': concentration,
        'bg_type': bg_type,
        'description': description,
        'filename': os.path.basename(image_path)
    }

def main():
    print("=== Real Experiment Image Grad-CAM Visualization Generator ===")
    
    # 设置中文字体
    chinese_font = setup_chinese_font()
    use_chinese = chinese_font is not None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # 加载模型
        models = {}
        if os.path.exists('best_model.pth'):
            models['all'] = load_model_safe('best_model.pth', device)
        if os.path.exists('best_model_bg0.pth'):
            models['bg0'] = load_model_safe('best_model_bg0.pth', device)
        
        if not models:
            print("❌ No model files found")
            return
        
        # 查找真实实验图像
        print("\n=== Finding Real Experiment Images ===")
        experiment_images = find_real_experiment_images()
        
        if not experiment_images:
            print("❌ No real experiment images found")
            return
        
        print(f"\nFound {len(experiment_images)} real experiment images")
        
        # 为每个模型生成可视化
        for model_name, model in models.items():
            print(f"\n=== Generating {model_name} model visualization ===")
            
            # 根据模型类型过滤图像
            if model_name == 'bg0':
                # bg0模型只处理bg0图像
                filtered_images = [img for img in experiment_images if img['bg_type'] == 'bg0']
            else:
                # all模型处理所有图像
                filtered_images = experiment_images
            
            if not filtered_images:
                print(f"❌ No suitable images for {model_name} model")
                continue
            
            # 处理每个图像
            visualizations = []
            for image_info in filtered_images:
                viz = create_gradcam_visualization(model, model_name, image_info, transform, device, use_chinese)
                if viz:
                    visualizations.append(viz)
            
            if not visualizations:
                print(f"❌ No successful visualizations for {model_name} model")
                continue
            
            # 创建综合可视化图
            num_images = len(visualizations)
            fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
            
            # 如果只有一个图像，确保axes是二维的
            if num_images == 1:
                axes = axes.reshape(1, -1)
            
            for i, viz in enumerate(visualizations):
                # 原始图像
                axes[i, 0].imshow(viz['img_np'])
                if use_chinese:
                    axes[i, 0].set_title(f'原始图像\n{viz["description"]}')
                else:
                    axes[i, 0].set_title(f'Original Image\n{viz["description"]}')
                axes[i, 0].axis('off')
                
                # Grad-CAM热力图
                im = axes[i, 1].imshow(viz['cam_resized'], cmap='jet')
                if use_chinese:
                    axes[i, 1].set_title('Grad-CAM热力图')
                else:
                    axes[i, 1].set_title('Grad-CAM Heatmap')
                axes[i, 1].axis('off')
                
                # 添加颜色条
                if i == 0:  # 只在第一行添加颜色条
                    cbar = plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
                    if use_chinese:
                        cbar.set_label('激活强度', rotation=270, labelpad=15)
                    else:
                        cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
                
                # 叠加可视化
                axes[i, 2].imshow(viz['overlay'])
                if use_chinese:
                    axes[i, 2].set_title('叠加可视化')
                else:
                    axes[i, 2].set_title('Overlay Visualization')
                axes[i, 2].axis('off')
                
                # 预测信息
                if use_chinese:
                    axes[i, 3].text(0.1, 0.8, f'模型: {model_name}', fontsize=10, weight='bold')
                    axes[i, 3].text(0.1, 0.65, f'图像: {viz["filename"][:30]}...', fontsize=8)
                    axes[i, 3].text(0.1, 0.5, f'真实浓度: {viz["concentration"]:.1f}', fontsize=10)
                    axes[i, 3].text(0.1, 0.35, f'预测浓度: {viz["pred_value"]:.1f}', fontsize=10)
                    axes[i, 3].text(0.1, 0.2, f'误差: {abs(viz["pred_value"] - viz["concentration"]):.1f}', fontsize=10)
                    axes[i, 3].text(0.1, 0.05, f'类型: {viz["bg_type"]}', fontsize=10)
                else:
                    axes[i, 3].text(0.1, 0.8, f'Model: {model_name}', fontsize=10, weight='bold')
                    axes[i, 3].text(0.1, 0.65, f'Image: {viz["filename"][:30]}...', fontsize=8)
                    axes[i, 3].text(0.1, 0.5, f'True Conc: {viz["concentration"]:.1f}', fontsize=10)
                    axes[i, 3].text(0.1, 0.35, f'Pred Conc: {viz["pred_value"]:.1f}', fontsize=10)
                    axes[i, 3].text(0.1, 0.2, f'Error: {abs(viz["pred_value"] - viz["concentration"]):.1f}', fontsize=10)
                    axes[i, 3].text(0.1, 0.05, f'Type: {viz["bg_type"]}', fontsize=10)
                axes[i, 3].set_xlim(0, 1)
                axes[i, 3].set_ylim(0, 1)
                axes[i, 3].axis('off')
            
            if use_chinese:
                plt.suptitle(f'{model_name}模型 - 真实实验图像Grad-CAM可视化', fontsize=16)
            else:
                plt.suptitle(f'{model_name} Model - Real Experiment Image Grad-CAM Visualization', fontsize=16)
            plt.tight_layout()
            
            # 保存图像
            output_path = f'real_experiment_gradcam_{model_name}_fixed_font.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved: {output_path}")
        
        print(f"\n=== Real experiment image visualization completed ===")
        
        # 创建Grad-CAM解读说明图
        create_gradcam_explanation()
        
    except Exception as e:
        print(f"❌ Program execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def create_gradcam_explanation():
    """创建Grad-CAM解读说明图"""
    print("\n=== Creating Grad-CAM Explanation ===")
    
    # 创建示例热力图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 创建不同激活强度的示例
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # 示例1: 高激活区域
    Z1 = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.1)
    im1 = axes[0, 0].imshow(Z1, cmap='jet', extent=[0, 1, 0, 1])
    axes[0, 0].set_title('High Activation\n(Red = Important Features)', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    
    # 示例2: 低激活区域
    Z2 = 0.2 * np.ones_like(Z1)
    im2 = axes[0, 1].imshow(Z2, cmap='jet', extent=[0, 1, 0, 1])
    axes[0, 1].set_title('Low Activation\n(Blue = Less Important)', fontsize=12, weight='bold')
    axes[0, 1].axis('off')
    
    # 示例3: 混合激活
    Z3 = np.exp(-((X-0.3)**2 + (Y-0.7)**2) / 0.05) + 0.5 * np.exp(-((X-0.7)**2 + (Y-0.3)**2) / 0.08)
    im3 = axes[0, 2].imshow(Z3, cmap='jet', extent=[0, 1, 0, 1])
    axes[0, 2].set_title('Mixed Activation\n(Multiple Focus Areas)', fontsize=12, weight='bold')
    axes[0, 2].axis('off')
    
    # 颜色条说明
    cbar = plt.colorbar(im1, ax=axes[0, :], fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity (0=Low, 1=High)', rotation=270, labelpad=20, fontsize=12)
    
    # 下半部分：解读说明
    axes[1, 0].text(0.1, 0.9, 'Color Interpretation:', fontsize=14, weight='bold')
    axes[1, 0].text(0.1, 0.8, '• Red/Yellow: High importance', fontsize=12, color='red')
    axes[1, 0].text(0.1, 0.7, '• Green: Medium importance', fontsize=12, color='green')
    axes[1, 0].text(0.1, 0.6, '• Blue/Purple: Low importance', fontsize=12, color='blue')
    axes[1, 0].text(0.1, 0.4, 'Model Focus:', fontsize=14, weight='bold')
    axes[1, 0].text(0.1, 0.3, '• Red areas = Key features', fontsize=12)
    axes[1, 0].text(0.1, 0.2, '• Blue areas = Background', fontsize=12)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.1, 0.9, 'What Grad-CAM Shows:', fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.8, '• Where the model "looks"', fontsize=12)
    axes[1, 1].text(0.1, 0.7, '• Which pixels influence prediction', fontsize=12)
    axes[1, 1].text(0.1, 0.6, '• Feature importance map', fontsize=12)
    axes[1, 1].text(0.1, 0.4, 'For Sediment Detection:', fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.3, '• Red = Sediment particles', fontsize=12)
    axes[1, 1].text(0.1, 0.2, '• Blue = Clear water', fontsize=12)
    axes[1, 1].text(0.1, 0.1, '• Green = Intermediate turbidity', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    axes[1, 2].text(0.1, 0.9, 'Analysis Tips:', fontsize=14, weight='bold')
    axes[1, 2].text(0.1, 0.8, '• Check if red areas match', fontsize=12)
    axes[1, 2].text(0.1, 0.75, '  expected sediment locations', fontsize=12)
    axes[1, 2].text(0.1, 0.65, '• Scattered red = good detection', fontsize=12)
    axes[1, 2].text(0.1, 0.55, '• Concentrated red = possible', fontsize=12)
    axes[1, 2].text(0.1, 0.5, '  artifacts or noise', fontsize=12)
    axes[1, 2].text(0.1, 0.4, '• Blue dominance = model', fontsize=12)
    axes[1, 2].text(0.1, 0.35, '  not finding features', fontsize=12)
    axes[1, 2].text(0.1, 0.25, '• Compare different models', fontsize=12)
    axes[1, 2].text(0.1, 0.2, '  to see focus differences', fontsize=12)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.suptitle('Grad-CAM Visualization Interpretation Guide', fontsize=18, weight='bold')
    plt.tight_layout()
    
    # 保存解读说明图
    explanation_path = 'gradcam_interpretation_guide.png'
    plt.savefig(explanation_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved interpretation guide: {explanation_path}")

if __name__ == '__main__':
    main() 