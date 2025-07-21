# 云端VGG训练包

## 📋 包信息
- **创建时间**: 2025-06-06T10:56:58.697915
- **目标平台**: RTX 4090 GPU
- **预估大小**: 5-8 GB
- **训练目标**: 特征数据集VGG回归

## 🚀 快速开始

### 1. 上传到云端
```bash
# 将整个文件夹上传到云端服务器
scp -r cloud_vgg_training_package user@server:/path/to/training/
```

### 2. 环境准备
```bash
cd cloud_vgg_training_package
chmod +x start_training.sh
```

### 3. 开始训练

#### 全部6档训练
```bash
./start_training.sh
```

#### 单个6档训练
```bash
# VGG不补光20mW训练
python cloud_vgg_train.py bg0_20mw

# VGG补光100mW训练  
python cloud_vgg_train.py bg1_100mw

# 直接使用训练脚本
python train_with_feature_dataset.py --model_type vgg --bg_mode bg0_400mw
```

#### 断点续训功能
```bash
# 指定断点续训目录
python cloud_vgg_train.py bg0_20mw --resume feature_training_vgg_bg0-20mw_results_20250606_143022

# 自动检测最新训练目录续训
python cloud_vgg_train.py bg0_20mw  # 自动寻找 bg0_20mw 的最新检查点

# 直接使用训练脚本断点续训
python train_with_feature_dataset.py --model_type vgg --bg_mode bg0_20mw --resume ./training_dir
```

## 📊 配置说明

### GPU配置 (configs/gpu_config.json)
- **GPU型号**: RTX 4090 (24GB)
- **批次大小**: 256 (充分利用显存)
- **工作线程**: 8
- **混合精度**: 启用

### 训练配置 (configs/training_config.json)  
- **模型**: VGG16 + CBAM注意力
- **训练模式**: 6档细分 (bg0/bg1 × 20mw/100mw/400mw)
- **轮次**: 50
- **学习率**: 0.001
- **早停**: 15轮耐心值

### 数据配置 (configs/data_config.json)
- **数据集**: 特征数据集 (146,644张图片)
- **划分比例**: 训练70% / 验证15% / 测试15%
- **增强策略**: 最小增强(保持物理特征)

## 📁 文件结构
```
cloud_vgg_training_package/
├── src/                    # 核心代码
│   ├── vgg_regression.py   # VGG训练主脚本
│   ├── cbam.py            # CBAM注意力模块
│   ├── fast_dataset.py    # 快速数据集加载器
│   └── ...                # 其他工具模块
├── feature_dataset/        # 特征数据集
├── configs/               # 配置文件
├── cloud_vgg_train.py     # 云端训练启动器
├── requirements.txt       # Python依赖
├── start_training.sh      # 一键启动脚本
└── README.md             # 说明文档
```

## 🎯 优化特性
- ✅ RTX 4090专用配置优化
- ✅ 大批次训练(batch_size=256)
- ✅ 混合精度训练加速
- ✅ 多线程数据加载
- ✅ 早停机制防过拟合
- ✅ 精简依赖快速部署

## 📈 预期性能
- **显存使用**: ~20-22GB (充分利用4090)
- **训练速度**: ~2-3x 本地速度提升
- **收敛时间**: 预计15-25个epoch
- **最终精度**: R² > 0.95

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

## 📞 支持
如有问题，请检查训练日志或联系技术支持。
