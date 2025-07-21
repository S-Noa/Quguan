#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50回归模型
用于激光光斑浓度预测
"""

import torch
import torch.nn as nn
from torchvision import models

class ResNet50Regression(nn.Module):
    """基于ResNet50的回归模型"""
    
    def __init__(self, freeze_backbone=True, dropout_rate=0.5):
        super(ResNet50Regression, self).__init__()
        
        # 加载预训练ResNet50
        print("加载ResNet50预训练模型...", flush=True)
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        
        # 移除分类头，获取特征
        self.backbone.fc = nn.Identity()
        
        # 冻结backbone参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("ResNet50 backbone已冻结", flush=True)
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, 1)
        )
        
        # 初始化回归头权重
        self._initialize_regressor()
        
        # 打印模型统计
        self._print_model_stats()
    
    def _initialize_regressor(self):
        """初始化回归头权重"""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        print("回归头权重初始化完成", flush=True)
    
    def _print_model_stats(self):
        """打印模型统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"ResNet50模型统计:", flush=True)
        print(f"  总参数: {total_params:,}", flush=True)
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)", flush=True)
        print(f"  冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)", flush=True)
        
        model_size_mb = total_params * 4 / (1024 * 1024)
        print(f"  模型大小: {model_size_mb:.1f} MB", flush=True)
    
    def forward(self, x):
        # 特征提取
        features = self.backbone(x)
        
        # 回归预测
        concentration = self.regressor(features)
        
        return concentration

def create_resnet50_model(freeze_backbone=True, dropout_rate=0.5):
    """创建ResNet50回归模型的工厂函数"""
    return ResNet50Regression(freeze_backbone=freeze_backbone, dropout_rate=dropout_rate)

if __name__ == "__main__":
    # 测试模型
    model = create_resnet50_model()
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)
        print(f"测试输入: {test_input.shape}")
        print(f"测试输出: {output.shape}")
        print(f"输出值范围: {output.min():.3f} - {output.max():.3f}")
    
    print("ResNet50模型测试完成！") 