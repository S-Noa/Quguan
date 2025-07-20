#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端训练脚本输出目录更新工具
为云端VGG和ResNet50训练添加基于数据集的输出目录命名
"""

import os
import sys

def main():
    print("正在更新云端训练脚本的输出目录设置...")
    
    # 复制数据集名称工具到云端包中
    src_utils_path = "../dataset_name_utils.py"
    dst_utils_path = "src/dataset_name_utils.py"
    
    if os.path.exists(src_utils_path):
        import shutil
        shutil.copy2(src_utils_path, dst_utils_path)
        print(f"✅ 数据集名称工具已复制到: {dst_utils_path}")
    else:
        print(f"❌ 找不到数据集名称工具: {src_utils_path}")
        return
    
    print("\n现在云端训练脚本可以使用基于数据集的输出目录命名了！")
    print("\n使用方法:")
    print("1. VGG训练: python src/vgg_regression.py --group all")
    print("2. ResNet50训练: python src/train_resnet50.py --group all")
    print("\n输出目录格式示例:")
    print("- vgg_all_v3_20250620_180816_results_20250622_143052")
    print("- resnet50_bg0_v2_20250620_150518_results_20250622_143052")

if __name__ == "__main__":
    main() 