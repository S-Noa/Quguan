#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端VGG训练启动脚本 - 4090 GPU优化版本
基于本地vgg_regression.py优化而来
"""

import sys
import os
import json

# 添加src目录到路径
sys.path.append('src')

def load_cloud_config():
    """加载云端配置"""
    with open('configs/gpu_config.json', 'r') as f:
        gpu_config = json.load(f)
    with open('configs/training_config.json', 'r') as f:
        training_config = json.load(f)
    with open('configs/data_config.json', 'r') as f:
        data_config = json.load(f)
    
    return gpu_config, training_config, data_config

def main():
    print("=== 云端6档细分训练启动 ===")
    print("=" * 60)
    
    # 解析命令行参数
    resume_dir = None
    selected_mode = None
    selected_model = None
    
    # 简单参数解析
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--resume' and i + 1 < len(args):
            resume_dir = args[i + 1]
            i += 2
        elif args[i] == '--model' and i + 1 < len(args):
            selected_model = args[i + 1]
            i += 2
        elif args[i] in ['bg0_20mw', 'bg0_100mw', 'bg0_400mw', 'bg1_20mw', 'bg1_100mw', 'bg1_400mw']:
            selected_mode = args[i]
            i += 1
        else:
            i += 1
    
    # 加载配置
    gpu_config, training_config, data_config = load_cloud_config()
    
    print(f"GPU配置: {gpu_config['gpu_model']} ({gpu_config['memory_gb']}GB)")
    print(f"批次大小: {gpu_config['recommended_batch_size']}")
    print(f"训练轮次: {training_config['epochs']}")
    
    if resume_dir:
        print(f"断点续训模式: {resume_dir}")
    
    # 6档训练模式
    training_modes = [
        'bg0_20mw', 'bg0_100mw', 'bg0_400mw',
        'bg1_20mw', 'bg1_100mw', 'bg1_400mw'
    ]
    
    # 3种模型类型
    model_types = ['vgg', 'resnet50']  # cnn通常不在云端训练
    
    print(f"\n云端训练计划:")
    for model_type in model_types:
    for i, mode in enumerate(training_modes, 1):
            print(f"   {model_type.upper()} - {mode}")
    
    # 检查命令行参数是否指定了特定模式
    if selected_mode and selected_model:
        print(f"\n>>> 执行指定训练: {selected_model.upper()} - {selected_mode}")
        success = run_single_training(selected_mode, selected_model, gpu_config, training_config, data_config, resume_dir)
        if success:
            print(f"\n=== {selected_model.upper()} - {selected_mode} 训练成功完成！ ===")
        else:
            print(f"\n!!! {selected_model.upper()} - {selected_mode} 训练失败！ !!!")
            sys.exit(1)
    elif selected_mode:
        # 如果只指定了模式，默认使用VGG模型
        print(f"\n>>> 执行指定模式 (默认VGG): {selected_mode}")
        success = run_single_training(selected_mode, 'vgg', gpu_config, training_config, data_config, resume_dir)
        if success:
            print(f"\n=== VGG - {selected_mode} 训练成功完成！ ===")
        else:
            print(f"\n!!! VGG - {selected_mode} 训练失败！ !!!")
            sys.exit(1)
    else:
        print(f"\n>>> 执行全部模型6档训练...")
        success_count = 0
        failed_trainings = []
        total_trainings = len(model_types) * len(training_modes)
        
        for model_type in model_types:
        for i, mode in enumerate(training_modes, 1):
                training_id = f"{model_type.upper()}-{mode}"
                print(f"\n{'='*60}")
                print(f"进度: {success_count + len(failed_trainings) + 1}/{total_trainings} - {training_id}")
                success = run_single_training(mode, model_type, gpu_config, training_config, data_config, resume_dir)
            if success:
                success_count += 1
            else:
                    failed_trainings.append(training_id)
        
        print(f"\n=== 云端训练总结 ===")
        print(f"   成功: {success_count}/{total_trainings}")
        if failed_trainings:
            print(f"   失败: {', '.join(failed_trainings)}")
        else:
            print(f"   全部训练成功完成！")

def run_single_training(bg_mode, model_type, gpu_config, training_config, data_config, resume_dir=None):
    """运行单个训练任务"""
    import subprocess
    import sys
    import glob
    
    print(f">>> 开始训练: {model_type.upper()} - {bg_mode}")
    
    # 构建基础训练命令
    cmd = [
        sys.executable, 'train_with_feature_dataset.py',
        '--model_type', model_type,
        '--bg_mode', bg_mode,  # 6档细分模式
        '--feature_dataset_path', data_config['dataset_path'],
        '--batch_size', str(gpu_config['recommended_batch_size']),
        '--epochs', str(training_config['epochs']),
        '--learning_rate', str(training_config['learning_rate']),
        '--weight_decay', str(training_config['weight_decay']),
        '--optimizer', 'adam',
        '--scheduler', 'step',
        '--step_size', '30',
        '--gamma', '0.1',
        '--save_checkpoint_every', str(training_config['save_checkpoint_every'])
    ]
    
    # 检查是否需要断点续训
    if resume_dir:
        print(f">>> 指定断点续训目录: {resume_dir}")
        cmd.extend(['--resume', resume_dir])
    else:
        # 自动检测断点续训目录
        pattern = f"feature_training_{model_type}_{bg_mode.replace('_', '-')}_results_*"
        existing_dirs = glob.glob(pattern)
        
        if existing_dirs:
            # 找到最新的训练目录
            latest_dir = max(existing_dirs, key=lambda x: os.path.getctime(x))
            checkpoint_path = os.path.join(latest_dir, 'checkpoint_latest.pth')
            
            if os.path.exists(checkpoint_path):
                print(f">>> 检测到断点续训目录: {latest_dir}")
                cmd.extend(['--resume', latest_dir])
            else:
                print(f">>> 未找到有效检查点，开始新训练")
        else:
            print(f">>> 未找到历史训练，开始新训练")
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 使用subprocess运行训练，确保参数正确传递
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"=== {model_type.upper()} - {bg_mode} 训练完成 ===")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! {model_type.upper()} - {bg_mode} 训练失败: 返回码 {e.returncode} !!!")
        return False
    except Exception as e:
        print(f"!!! {model_type.upper()} - {bg_mode} 训练失败: {e} !!!")
        return False

if __name__ == "__main__":
    main()
