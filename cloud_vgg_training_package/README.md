# 云端训练包（部分信息未更新）

## 📋 包信息
- **创建时间**: 2025-06-06T10:56:58.697915
- **目标平台**: RTX 4090 GPU
- **训练目标**: 特征数据集VGG回归

## 🚀 快速开始

### 1. 上传到云端
```bash
# 将整个文件夹上传到云端服务器 具体命令根据云端平台确定
scp -r cloud_vgg_training_package user@server:/path/to/training/
```

### 2. 环境准备
```bash
cd cloud_vgg_training_package
chmod +x start_training.sh
```

### 3. 开始训练

#### 查看参数
```bash
# 查看训练脚本参数
python train_with_feature_dataset.py -h
```

#### 训练示例
```bash
# 训练命令示例
python cloud_vgg_training_package/train_with_feature_dataset.py \
    --model_type resnet50 \
    --bg_mode bg1_400mw \
    --epochs 10 \
    --progressive_weighting \
    --warmup_epochs 3 \
    --smooth_scale_factor 0.01 \
```

## 📊 配置说明

### GPU配置 (configs/gpu_config.json)
- **GPU型号**: RTX 4090 (24GB)
- **批次大小**: 256
- **工作线程**: 8
- **混合精度**: 启用

### 训练配置 (configs/training_config.json)  
- **模型**: VGG16 + CBAM注意力
- **训练模式**: 6档细分 (bg0/bg1 × 20mw/100mw/400mw)
- **轮次**: 50
- **学习率**: 0.001
- **早停**: 15轮耐心值

## 🔧 故障排除

### 显存不足
```bash
# 降低batch_size
vim configs/gpu_config.json
# 修改 "recommended_batch_size": 128
```

### 依赖安装失败
```bash
# 手动安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
