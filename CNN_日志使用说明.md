# CNN训练日志功能使用说明

## 📋 概述

已为CNN训练脚本 `train.py` 添加了完整的日志记录功能，可以详细记录训练过程中的所有信息，包括系统状态、模型参数、训练进度、性能指标等。

## 🚀 主要功能

### 1. **自动日志文件生成**
- 每次运行自动生成带时间戳的日志文件
- 文件命名格式：`cnn_training_{group}_{YYYYMMDD_HHMMSS}.log`
- 同时输出到控制台和文件

### 2. **详细系统信息记录**
```
=== 系统信息 ===
Python版本: 3.9.13
PyTorch版本: 2.1.1+cu118
CUDA可用: True
CUDA版本: 11.8
GPU数量: 1
GPU 0: NVIDIA GeForce RTX 3060 Ti
GPU 0 显存: 8.0 GB
```

### 3. **完整训练过程记录**
- **模型信息**：参数数量、模型大小、设备信息
- **数据集信息**：样本数量、浓度分布、数据划分
- **训练配置**：学习率、批次大小、优化器、损失函数
- **训练进度**：每个epoch的损失、R²分数、耗时
- **性能监控**：GPU内存使用、批次处理时间
- **评估结果**：MSE、RMSE、MAE、R²等详细指标

### 4. **错误处理和异常记录**
- 自动捕获并记录训练过程中的异常
- 详细的错误堆栈信息
- 优雅的错误恢复机制

## 📖 使用方法

### 基本用法
```bash
# 训练全部数据（生成 cnn_training_all_YYYYMMDD_HHMMSS.log）
python train.py --group all

# 训练bg0数据（生成 cnn_training_bg0_YYYYMMDD_HHMMSS.log）
python train.py --group bg0

# 训练bg1数据（生成 cnn_training_bg1_YYYYMMDD_HHMMSS.log）
python train.py --group bg1

# 训练所有分组（生成 cnn_training_allgroup_YYYYMMDD_HHMMSS.log）
python train.py --group allgroup
```

### 高级参数
```bash
# 自定义训练参数
python train.py --group bg0 --epochs 20 --batch_size 32 --lr 0.0005 --data_path "D:/your/data/path"
```

### 参数说明
- `--group`: 训练数据分组 (all/bg0/bg1/allgroup)
- `--epochs`: 训练轮数 (默认: 15)
- `--batch_size`: 批次大小 (默认: 16)
- `--lr`: 学习率 (默认: 0.001)
- `--data_path`: 数据集路径 (默认: "D:\2025年实验照片")

## 📊 日志内容示例

### 训练开始
```
2025-05-29 09:04:37 - CNN_Training - INFO - === CNN训练程序启动 ===
2025-05-29 09:04:37 - CNN_Training - INFO - 命令行参数: {'group': 'bg0', 'epochs': 15, 'batch_size': 16, 'lr': 0.001, 'data_path': 'D:\\2025年实验照片'}
2025-05-29 09:04:37 - CNN_Training - INFO - 日志文件: cnn_training_bg0_20250529_090437.log
2025-05-29 09:04:37 - CNN_Training - INFO - 使用设备: cuda
```

### 数据集信息
```
2025-05-29 09:04:38 - CNN_Training - INFO - === bg0数据集信息 ===
2025-05-29 09:04:38 - CNN_Training - INFO - 数据集大小: 76037 张图像
2025-05-29 09:04:38 - CNN_Training - INFO - 浓度范围: 0.0 - 500.0
2025-05-29 09:04:38 - CNN_Training - INFO - 平均浓度: 250.15
2025-05-29 09:04:38 - CNN_Training - INFO - 浓度标准差: 144.34
```

### 模型信息
```
2025-05-29 09:04:39 - CNN_Training - INFO - === 模型信息 ===
2025-05-29 09:04:39 - CNN_Training - INFO - 模型总参数: 25,690,113
2025-05-29 09:04:39 - CNN_Training - INFO - 可训练参数: 25,690,113
2025-05-29 09:04:39 - CNN_Training - INFO - 模型大小: 98.0 MB
2025-05-29 09:04:39 - CNN_Training - INFO - 模型设备: cuda
```

### 训练过程
```
2025-05-29 09:04:40 - CNN_Training - INFO - === 开始训练 ===
2025-05-29 09:04:40 - CNN_Training - INFO - 训练轮数: 15
2025-05-29 09:04:40 - CNN_Training - INFO - 批次大小: 16
2025-05-29 09:04:40 - CNN_Training - INFO - 训练批次数: 3322
2025-05-29 09:04:40 - CNN_Training - INFO - 验证批次数: 712

2025-05-29 09:04:45 - CNN_Training - INFO - --- Epoch 1/15 ---
2025-05-29 09:04:50 - CNN_Training - INFO - [Epoch 1, Batch 10] 训练损失: 45234.567890, 批次耗时: 0.85s
2025-05-29 09:04:50 - CNN_Training - INFO - GPU内存使用: 2.34GB / 3.12GB
```

### 评估结果
```
2025-05-29 09:15:23 - CNN_Training - INFO - === 评估结果 ===
2025-05-29 09:15:23 - CNN_Training - INFO - 测试样本数: 11406
2025-05-29 09:15:23 - CNN_Training - INFO - 均方误差 (MSE): 1234.567890
2025-05-29 09:15:23 - CNN_Training - INFO - 均方根误差 (RMSE): 35.136
2025-05-29 09:15:23 - CNN_Training - INFO - 平均绝对误差 (MAE): 28.945
2025-05-29 09:15:23 - CNN_Training - INFO - R² 分数: 0.856789
2025-05-29 09:15:23 - CNN_Training - INFO - 评估耗时: 12.3s
```

## 📁 生成的文件

### 日志文件
- `cnn_training_{group}_{timestamp}.log` - 详细训练日志
- 包含完整的训练过程记录
- UTF-8编码，支持中文

### 训练历史文件
- `training_history_{group}.npz` - 训练历史数据
- 包含训练损失、验证损失、R²分数的数组
- 可用于后续分析和可视化

### 可视化文件
- `cnn_prediction_plot_epoch_{N}_{group}.png` - 每5个epoch的预测图
- `cnn_final_prediction_plot_{group}.png` - 最终测试结果图
- `cnn_grad_cam_{group}.png` - Grad-CAM可视化图

## 🔧 日志分析

### 查看日志文件
```bash
# 查看最新的日志文件
ls -lt cnn_training_*.log | head -1

# 实时监控训练日志
tail -f cnn_training_bg0_20250529_090437.log

# 搜索特定信息
grep "R² 分数" cnn_training_bg0_20250529_090437.log
grep "ERROR" cnn_training_bg0_20250529_090437.log
```

### 性能分析
日志中包含详细的性能指标，可以用于：
- 分析训练收敛情况
- 监控GPU内存使用
- 识别性能瓶颈
- 比较不同配置的效果

## ⚠️ 注意事项

1. **磁盘空间**：日志文件可能较大，请确保有足够的磁盘空间
2. **编码问题**：日志文件使用UTF-8编码，确保文本编辑器支持
3. **权限问题**：确保程序有写入当前目录的权限
4. **日志清理**：定期清理旧的日志文件以节省空间

## 🎯 使用建议

1. **实验记录**：每次实验都会生成独立的日志文件，便于对比分析
2. **错误排查**：出现问题时，日志文件是最好的调试工具
3. **性能优化**：通过日志分析找出训练瓶颈
4. **结果复现**：日志记录了完整的训练配置，便于结果复现

现在你的CNN训练过程将被完整记录，便于分析和调试！🎉 