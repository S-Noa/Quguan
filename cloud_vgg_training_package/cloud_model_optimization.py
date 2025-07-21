#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端模型优化脚本
解决VGG+CBAM和ResNet50表现不如传统CNN的问题
"""

import os
import json
import torch
import numpy as np
from datetime import datetime

def analyze_performance_gap():
    """分析性能差距的可能原因"""
    print("=" * 60)
    print("云端模型性能诊断分析")
    print("=" * 60)
    
    issues = {
        "数据不一致": [
            "云端使用特征数据集 vs 本地使用原始数据集",
            "预处理方式不同（归一化、缩放）",
            "图像尺寸可能不一致",
            "数据增强策略差异"
        ],
        "模型配置": [
            "预训练权重与激光光斑数据领域差异大",
            "冻结策略过于保守，限制了特征学习",
            "学习率设置不当，收敛不充分",
            "正则化过强导致欠拟合"
        ],
        "训练策略": [
            "早停设置可能过于激进",
            "批次大小影响收敛稳定性",
            "优化器选择不适合大模型",
            "学习率调度器设置不当"
        ],
        "架构匹配": [
            "回归头设计可能过于简单",
            "CBAM注意力可能干扰特定任务学习",
            "预训练特征提取层与激光光斑检测不匹配"
        ]
    }
    
    for category, problem_list in issues.items():
        print(f"\n🔍 {category}问题:")
        for i, problem in enumerate(problem_list, 1):
            print(f"   {i}. {problem}")
    
    return issues

def create_optimized_vgg_config():
    """创建优化的VGG训练配置"""
    configs = {
        "渐进式解冻策略": {
            "description": "逐步解冻VGG层，避免特征层过于固化",
            "stage1": {
                "epochs": 30,
                "freeze_features": True,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "note": "只训练回归头"
            },
            "stage2": {
                "epochs": 50,
                "freeze_features": "partial",  # 解冻最后2个block
                "learning_rate": 5e-4,
                "batch_size": 16,
                "note": "微调高级特征"
            },
            "stage3": {
                "epochs": 70,
                "freeze_features": False,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "note": "端到端微调"
            }
        },
        
        "数据一致性优化": {
            "description": "确保与传统CNN使用相同的数据处理",
            "use_original_dataset": True,
            "image_size": 224,
            "normalization": "ImageNet标准",
            "augmentation": "保守增强策略"
        },
        
        "CBAM注意力优化": {
            "description": "优化CBAM注意力机制配置",
            "cbam_ratio": 8,  # 降低注意力复杂度
            "spatial_kernel": 3,  # 使用更小的空间卷积核
            "apply_stages": [3, 4, 5],  # 只在高级特征上应用
            "attention_dropout": 0.1
        },
        
        "回归头增强": {
            "description": "加强回归头设计",
            "layers": [
                {"type": "AdaptiveAvgPool2d", "size": (7, 7)},
                {"type": "Flatten"},
                {"type": "Linear", "in": 512*7*7, "out": 2048},
                {"type": "ReLU"},
                {"type": "Dropout", "p": 0.5},
                {"type": "Linear", "in": 2048, "out": 512},
                {"type": "ReLU"},
                {"type": "Dropout", "p": 0.3},
                {"type": "Linear", "in": 512, "out": 128},
                {"type": "ReLU"},
                {"type": "Dropout", "p": 0.1},
                {"type": "Linear", "in": 128, "out": 1}
            ]
        }
    }
    
    return configs

def create_optimized_resnet50_config():
    """创建优化的ResNet50训练配置"""
    configs = {
        "分层解冻策略": {
            "description": "ResNet50分层解冻，避免梯度消失",
            "stage1": {
                "epochs": 40,
                "freeze_backbone": True,
                "learning_rate": 5e-3,  # 较高学习率训练回归头
                "batch_size": 32,
                "note": "强化回归头训练"
            },
            "stage2": {
                "epochs": 60,
                "freeze_layers": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"],
                "learning_rate": 1e-3,
                "batch_size": 16,
                "note": "解冻高级特征层"
            },
            "stage3": {
                "epochs": 80,
                "freeze_layers": ["conv1", "bn1", "relu", "maxpool", "layer1"],
                "learning_rate": 1e-4,
                "batch_size": 16,
                "note": "进一步解冻中级特征"
            },
            "stage4": {
                "epochs": 100,
                "freeze_backbone": False,
                "learning_rate": 1e-5,
                "batch_size": 8,
                "note": "端到端精细调优"
            }
        },
        
        "领域适应优化": {
            "description": "针对激光光斑领域的特殊优化",
            "use_laser_specific_augmentation": True,
            "center_crop_probability": 0.8,  # 保持中心区域
            "rotation_limit": 5,  # 限制旋转角度
            "brightness_limit": 0.1,  # 小幅亮度调整
            "contrast_limit": 0.1
        },
        
        "增强回归头": {
            "description": "专门为激光光斑设计的回归头",
            "architecture": "deep_regressor",
            "layers": [
                {"type": "AdaptiveAvgPool2d", "size": (1, 1)},
                {"type": "Flatten"},
                {"type": "Linear", "in": 2048, "out": 1024},
                {"type": "BatchNorm1d", "features": 1024},
                {"type": "ReLU"},
                {"type": "Dropout", "p": 0.5},
                {"type": "Linear", "in": 1024, "out": 256},
                {"type": "BatchNorm1d", "features": 256},
                {"type": "ReLU"}, 
                {"type": "Dropout", "p": 0.3},
                {"type": "Linear", "in": 256, "out": 64},
                {"type": "ReLU"},
                {"type": "Dropout", "p": 0.1},
                {"type": "Linear", "in": 64, "out": 1}
            ]
        },
        
        "优化器配置": {
            "description": "针对大模型的优化器策略",
            "optimizer": "AdamW",
            "weight_decay": 1e-2,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "amsgrad": True
        }
    }
    
    return configs

def create_training_scripts():
    """创建优化的训练脚本"""
    
    # VGG优化脚本
    vgg_script = """#!/bin/bash
# VGG+CBAM 渐进式优化训练脚本

echo "开始VGG+CBAM优化训练..."

# 阶段1: 回归头训练
echo "阶段1: 训练回归头 (30 epochs)"
python train_with_feature_dataset.py \\
  --model_type vgg \\
  --bg_mode bg0_20mw \\
  --epochs 30 \\
  --batch_size 32 \\
  --learning_rate 1e-3 \\
  --weight_decay 1e-4 \\
  --scheduler step \\
  --step_size 10 \\
  --gamma 0.5 \\
  --early_stopping_patience 15

# 阶段2: 部分解冻
echo "阶段2: 部分解冻微调 (50 epochs)"
python train_with_feature_dataset.py \\
  --model_type vgg \\
  --bg_mode bg0_20mw \\
  --epochs 50 \\
  --batch_size 16 \\
  --learning_rate 5e-4 \\
  --weight_decay 5e-4 \\
  --scheduler cosine \\
  --early_stopping_patience 20 \\
  --resume [阶段1输出目录]

# 阶段3: 端到端微调
echo "阶段3: 端到端微调 (70 epochs)"
python train_with_feature_dataset.py \\
  --model_type vgg \\
  --bg_mode bg0_20mw \\
  --epochs 70 \\
  --batch_size 16 \\
  --learning_rate 1e-4 \\
  --weight_decay 1e-3 \\
  --scheduler cosine \\
  --early_stopping_patience 25 \\
  --resume [阶段2输出目录]

echo "VGG优化训练完成!"
"""
    
    # ResNet50优化脚本
    resnet_script = """#!/bin/bash
# ResNet50 分层解冻优化训练脚本

echo "开始ResNet50优化训练..."

# 阶段1: 强化回归头
echo "阶段1: 强化回归头训练 (40 epochs)"
python train_with_feature_dataset.py \\
  --model_type resnet50 \\
  --bg_mode bg0_20mw \\
  --epochs 40 \\
  --batch_size 32 \\
  --learning_rate 5e-3 \\
  --weight_decay 1e-4 \\
  --optimizer adamw \\
  --scheduler step \\
  --step_size 15 \\
  --gamma 0.3

# 阶段2: 解冻高级特征
echo "阶段2: 解冻高级特征 (60 epochs)"
python train_with_feature_dataset.py \\
  --model_type resnet50 \\
  --bg_mode bg0_20mw \\
  --epochs 60 \\
  --batch_size 16 \\
  --learning_rate 1e-3 \\
  --weight_decay 5e-4 \\
  --scheduler cosine \\
  --early_stopping_patience 20 \\
  --resume [阶段1输出目录]

# 阶段3: 解冻中级特征
echo "阶段3: 解冻中级特征 (80 epochs)"
python train_with_feature_dataset.py \\
  --model_type resnet50 \\
  --bg_mode bg0_20mw \\
  --epochs 80 \\
  --batch_size 16 \\
  --learning_rate 1e-4 \\
  --weight_decay 1e-3 \\
  --scheduler cosine \\
  --early_stopping_patience 25 \\
  --resume [阶段2输出目录]

# 阶段4: 端到端精调
echo "阶段4: 端到端精细调优 (100 epochs)"
python train_with_feature_dataset.py \\
  --model_type resnet50 \\
  --bg_mode bg0_20mw \\
  --epochs 100 \\
  --batch_size 8 \\
  --learning_rate 1e-5 \\
  --weight_decay 1e-2 \\
  --scheduler cosine \\
  --early_stopping_patience 30 \\
  --resume [阶段3输出目录]

echo "ResNet50优化训练完成!"
"""
    
    return vgg_script, resnet_script

def create_data_consistency_fix():
    """创建数据一致性修复方案"""
    
    consistency_issues = {
        "数据源差异": {
            "问题": "云端使用特征数据集，本地使用原始数据集",
            "解决方案": [
                "云端也使用原始数据集进行训练",
                "或者本地也使用特征数据集训练传统CNN对比",
                "统一数据预处理管道"
            ]
        },
        
        "图像预处理": {
            "问题": "预处理方式可能不一致",
            "标准化配置": {
                "mean": [0.485, 0.456, 0.406],  # ImageNet标准
                "std": [0.229, 0.224, 0.225],
                "size": 224,
                "interpolation": "BILINEAR"
            }
        },
        
        "数据增强": {
            "传统CNN增强": "基础增强（翻转、小幅旋转）",
            "预训练模型增强": "保守增强（避免破坏预训练特征）",
            "统一增强策略": {
                "RandomHorizontalFlip": 0.5,
                "RandomRotation": 5,  # 小角度旋转
                "ColorJitter": {
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "saturation": 0.1,
                    "hue": 0.05
                }
            }
        }
    }
    
    return consistency_issues

def generate_optimization_report():
    """生成优化建议报告"""
    
    print("\n" + "=" * 60)
    print("云端模型优化建议报告")
    print("=" * 60)
    
    # 分析问题
    issues = analyze_performance_gap()
    
    # 优化配置
    vgg_config = create_optimized_vgg_config()
    resnet_config = create_optimized_resnet50_config()
    
    # 数据一致性
    data_fix = create_data_consistency_fix()
    
    # 训练脚本
    vgg_script, resnet_script = create_training_scripts()
    
    # 生成报告
    report = {
        "生成时间": datetime.now().isoformat(),
        "问题诊断": issues,
        "VGG优化配置": vgg_config,
        "ResNet50优化配置": resnet_config,
        "数据一致性修复": data_fix,
        "immediate_actions": [
            "1. 确保云端和本地使用相同的数据集",
            "2. 统一图像预处理管道",
            "3. 采用渐进式解冻训练策略",
            "4. 增强回归头设计",
            "5. 调整学习率和正则化参数",
            "6. 延长训练时间，避免过早收敛"
        ]
    }
    
    # 保存配置文件
    with open("cloud_model_optimization_config.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 保存训练脚本
    with open("optimize_vgg_training.sh", "w") as f:
        f.write(vgg_script)
    
    with open("optimize_resnet50_training.sh", "w") as f:
        f.write(resnet_script)
    
    print("\n📋 优化建议总结:")
    print("1. 🔄 数据一致性: 确保云端和本地使用相同数据处理")
    print("2. 📈 渐进式训练: 分阶段解冻预训练层")
    print("3. 🎯 回归头增强: 设计更深层的回归网络")
    print("4. ⚙️ 参数调优: 针对大模型调整超参数")
    print("5. ⏰ 训练时长: 增加训练epoch，避免欠拟合")
    print("6. 🔍 监控对比: 与传统CNN使用相同评估指标")
    
    print(f"\n📄 详细配置已保存到:")
    print(f"   - cloud_model_optimization_config.json")
    print(f"   - optimize_vgg_training.sh")
    print(f"   - optimize_resnet50_training.sh")
    
    return report

def main():
    """主函数"""
    print("云端模型性能优化分析")
    
    # 生成优化报告
    report = generate_optimization_report()
    
    print("\n" + "=" * 60)
    print("🚀 立即可执行的优化步骤:")
    print("=" * 60)
    
    for i, action in enumerate(report["immediate_actions"], 1):
        print(f"{action}")
    
    print(f"\n💡 建议优先级:")
    print(f"1. 【高】数据一致性修复 - 可能是主要问题")
    print(f"2. 【高】渐进式解冻训练 - 预训练模型必需")
    print(f"3. 【中】回归头增强 - 提升拟合能力")
    print(f"4. 【中】超参数调优 - 匹配模型复杂度")
    print(f"5. 【低】训练时长调整 - 避免欠拟合")
    
    print(f"\n🎯 预期效果:")
    print(f"   经过优化后，VGG和ResNet50应该能达到或超过传统CNN的性能")
    print(f"   如果仍然不如传统CNN，说明预训练特征可能不适合此任务")

if __name__ == "__main__":
    main() 