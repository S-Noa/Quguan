#!/usr/bin/env python3
"""
高级CNN模型架构
包含注意力机制、多尺度特征提取等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class CBAM(nn.Module):
    """卷积块注意力模块 (CBAM)"""
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class LaserSpotFocusedAttention(nn.Module):
    """专门针对激光光斑的注意力机制"""
    
    def __init__(self, in_channels):
        super(LaserSpotFocusedAttention, self).__init__()
        self.conv_green = nn.Conv2d(in_channels, 1, 1)  # 绿色通道增强
        self.conv_center = nn.Conv2d(in_channels, 1, 1)  # 中心区域增强
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 绿色通道注意力（假设输入是RGB）
        green_attention = self.sigmoid(self.conv_green(x))
        
        # 中心区域注意力
        center_attention = self._generate_center_mask(x.shape[-2:], x.device)
        center_attention = center_attention.unsqueeze(0).unsqueeze(0).expand_as(green_attention)
        
        # 结合两种注意力
        combined_attention = green_attention * center_attention
        
        return x * combined_attention
    
    def _generate_center_mask(self, size, device):
        """生成中心区域掩码"""
        h, w = size
        y, x = torch.meshgrid(torch.arange(h, device=device), 
                             torch.arange(w, device=device), indexing='ij')
        
        center_y, center_x = h // 2, w // 2
        distance = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_distance = torch.sqrt(torch.tensor(center_y ** 2 + center_x ** 2, device=device))
        
        # 高斯式衰减
        mask = torch.exp(-0.5 * (distance / (max_distance * 0.3)) ** 2)
        return mask

class MultiScaleConvBlock(nn.Module):
    """多尺度卷积块"""
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvBlock, self).__init__()
        
        # 不同尺度的卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out1 = self.conv1x1(x)
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)
        out7 = self.conv7x7(x)
        
        out = torch.cat([out1, out3, out5, out7], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AdvancedLaserSpotCNN(nn.Module):
    """高级激光光斑CNN模型"""
    
    def __init__(self, num_features=512, use_attention=True, use_multiscale=True, use_laser_attention=True):
        super(AdvancedLaserSpotCNN, self).__init__()
        
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        self.use_laser_attention = use_laser_attention  # 新增参数，控制激光光斑专用注意力
        
        # 特征提取层
        if use_multiscale:
            self.conv1 = MultiScaleConvBlock(3, 64)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 注意力机制
        if use_attention:
            # 激光光斑专用注意力（可选）
            if use_laser_attention:
                self.laser_attention = LaserSpotFocusedAttention(512)
            # 通用CBAM注意力
            self.cbam = CBAM(512)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # 回归输出
        )
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(num_features, num_features)
        )
        
        # 用于GradCAM的最后卷积层
        self.last_conv = self.layer4
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        # 特征提取
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 注意力机制
        if self.use_attention:
            # 激光光斑专用注意力（仅在use_laser_attention=True时启用）
            if self.use_laser_attention:
                x = self.laser_attention(x)
            # 通用CBAM注意力
            x = self.cbam(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 浓度预测
        concentration = self.classifier(x)
        
        # 特征提取
        features = self.feature_extractor(x)
        
        return concentration, features

class LaserSpotTransformer(nn.Module):
    """基于Transformer的激光光斑分析模型"""
    
    def __init__(self, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        super(LaserSpotTransformer, self).__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CNN特征提取器
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 512, 2),
        )
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(512, embed_dim, patch_size, patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))  # 14x14 patches
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # CNN特征提取
        x = self.cnn_backbone(x)
        
        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, embed_dim
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 浓度预测
        concentration = self.classifier(x)
        
        # 特征（使用transformer的输出）
        features = x
        
        return concentration, features

def create_advanced_model(model_type='advanced_cnn', **kwargs):
    """创建高级模型的工厂函数"""
    
    if model_type == 'advanced_cnn':
        return AdvancedLaserSpotCNN(**kwargs)
    elif model_type == 'transformer':
        return LaserSpotTransformer(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 高级CNN模型
    model_cnn = create_advanced_model('advanced_cnn', use_attention=True, use_multiscale=True)
    model_cnn = model_cnn.to(device)
    
    # Transformer模型
    model_transformer = create_advanced_model('transformer')
    model_transformer = model_transformer.to(device)
    
    # 测试输入
    x = torch.randn(2, 3, 224, 224).to(device)
    
    # 测试CNN
    print("测试高级CNN模型:")
    with torch.no_grad():
        conc, feat = model_cnn(x)
        print(f"输入形状: {x.shape}")
        print(f"浓度预测形状: {conc.shape}")
        print(f"特征形状: {feat.shape}")
    
    # 测试Transformer
    print("\n测试Transformer模型:")
    with torch.no_grad():
        conc, feat = model_transformer(x)
        print(f"输入形状: {x.shape}")
        print(f"浓度预测形状: {conc.shape}")
        print(f"特征形状: {feat.shape}")
    
    # 参数统计
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"高级CNN: {count_parameters(model_cnn):,}")
    print(f"Transformer: {count_parameters(model_transformer):,}") 