#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字体设置工具模块
用于解决matplotlib中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import os
import platform

def setup_chinese_font():
    """
    设置matplotlib的中文字体
    
    Returns:
        str or None: 成功设置的字体名称，如果失败则返回None
    """
    try:
        # 检测操作系统
        system = platform.system()
        
        # 根据不同系统尝试不同的中文字体
        if system == "Windows":
            chinese_fonts = [
                'SimHei',           # 黑体
                'Microsoft YaHei',  # 微软雅黑
                'SimSun',           # 宋体
                'KaiTi',            # 楷体
                'FangSong',         # 仿宋
            ]
        elif system == "Linux":
            chinese_fonts = [
                'WenQuanYi Micro Hei',  # 文泉驿微米黑
                'WenQuanYi Zen Hei',    # 文泉驿正黑
                'Noto Sans CJK SC',     # Google Noto字体
                'Source Han Sans CN',   # 思源黑体
                'AR PL UMing CN',       # AR PL字体
                'AR PL UKai CN',
                'SimHei',               # 如果安装了Windows字体
                'Microsoft YaHei',
            ]
        elif system == "Darwin":  # macOS
            chinese_fonts = [
                'PingFang SC',          # 苹果苹方
                'Hiragino Sans GB',     # 冬青黑体
                'STHeiti',              # 华文黑体
                'SimHei',
                'Microsoft YaHei',
            ]
        else:
            chinese_fonts = ['SimHei', 'Microsoft YaHei']
        
        # 添加通用备选字体
        chinese_fonts.extend(['DejaVu Sans', 'Arial', 'Liberation Sans'])
        
        # 获取系统可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        print(f"检测到操作系统: {system}")
        print(f"系统可用字体数量: {len(available_fonts)}")
        
        # 尝试找到合适的中文字体
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用字体: {font}")
                return font
        
        # 如果没有找到中文字体，尝试强制设置
        print("❌ 未找到标准中文字体，尝试强制设置...")
        
        # 在Linux上尝试直接设置字体路径
        if system == "Linux":
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        # 尝试添加字体路径
                        fm.fontManager.addfont(font_path)
                        # 重新获取字体列表
                        available_fonts = [f.name for f in fm.fontManager.ttflist]
                        print(f"添加字体文件: {font_path}")
                        break
                    except Exception as e:
                        print(f"添加字体失败 {font_path}: {e}")
        
        # 再次尝试设置中文字体
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用字体: {font}")
                return font
        
        # 最后的备选方案：使用英文字体并返回None
        print("❌ 未找到中文字体，将使用英文标签")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return None
        
    except Exception as e:
        print(f"字体设置失败: {e}")
        # 设置默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return None

def get_labels(use_chinese=True):
    """
    根据字体支持情况返回相应的标签
    
    Args:
        use_chinese (bool): 是否使用中文标签
        
    Returns:
        dict: 包含各种标签的字典
    """
    if use_chinese:
        return {
            'true_concentration': '真实浓度值',
            'predicted_concentration': '预测浓度值',
            'test_results': '测试集预测结果',
            'all_data': '全部数据',
            'original_image': '原始图像',
            'gradcam_heatmap': 'Grad-CAM热力图',
            'overlay_visualization': '叠加可视化',
            'activation_intensity': '激活强度',
            'model': '模型',
            'true_conc': '真实浓度',
            'pred_conc': '预测浓度',
            'error': '误差',
            'type': '类型',
            'epoch': 'Epoch',
            'prediction_vs_true': '预测值 vs 真实值',
            'final_results': '最终结果',
        }
    else:
        return {
            'true_concentration': 'True Concentration',
            'predicted_concentration': 'Predicted Concentration',
            'test_results': 'Test Set Prediction Results',
            'all_data': 'All Data',
            'original_image': 'Original Image',
            'gradcam_heatmap': 'Grad-CAM Heatmap',
            'overlay_visualization': 'Overlay Visualization',
            'activation_intensity': 'Activation Intensity',
            'model': 'Model',
            'true_conc': 'True Conc',
            'pred_conc': 'Pred Conc',
            'error': 'Error',
            'type': 'Type',
            'epoch': 'Epoch',
            'prediction_vs_true': 'Prediction vs True',
            'final_results': 'Final Results',
        }

def suppress_font_warnings():
    """抑制matplotlib字体相关的警告"""
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', message='Glyph.*missing from current font')

def install_chinese_fonts_linux():
    """在Linux系统上安装中文字体的建议"""
    system = platform.system()
    if system == "Linux":
        print("\n=== Linux中文字体安装建议 ===")
        print("如果需要中文支持，请运行以下命令安装字体：")
        print("Ubuntu/Debian:")
        print("  sudo apt-get update")
        print("  sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei")
        print("  sudo apt-get install fonts-noto-cjk")
        print("\nCentOS/RHEL:")
        print("  sudo yum install wqy-microhei-fonts wqy-zenhei-fonts")
        print("  sudo yum install google-noto-cjk-fonts")
        print("\n安装后重启Python程序即可支持中文显示")

# 自动设置字体（导入时执行）
def auto_setup_font():
    """自动设置字体并返回是否支持中文"""
    suppress_font_warnings()
    font_name = setup_chinese_font()
    
    # 如果是Linux且没有中文字体，显示安装建议
    if font_name is None and platform.system() == "Linux":
        install_chinese_fonts_linux()
    
    return font_name is not None

# 模块级变量，记录是否支持中文
CHINESE_SUPPORTED = auto_setup_font() 