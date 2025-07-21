#!/bin/bash
# 云端VGG训练启动脚本

echo "🚀 开始云端VGG训练"
echo "============================"

# 检查GPU
echo "🔍 检查GPU状态:"
nvidia-smi

# 安装依赖
echo "📦 安装Python依赖:"
pip install -r requirements.txt

# 开始训练
echo "🏃 开始6档细分VGG训练:"
echo "   执行全部6档: bg0_20mw, bg0_100mw, bg0_400mw, bg1_20mw, bg1_100mw, bg1_400mw"
python cloud_vgg_train.py

echo "✅ 训练完成"
