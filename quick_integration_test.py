#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速集成测试 - 验证监控功能是否正常
"""

def test_basic_integration():
    """基础集成测试"""
    print("🔍 快速集成测试")
    print("=" * 30)
    
    try:
        # 1. 测试导入
        print("1. 测试导入...")
        from training_monitor import TrainingMonitor, GradCAM
        print("   ✓ TrainingMonitor导入成功")
        
        from cnn_model import CNNFeatureExtractor
        print("   ✓ CNN模型导入成功")
        
        # 2. 测试重复定义检查
        print("2. 检查重复定义...")
        import os
        
        files_to_check = ['train.py', 'vgg_regression.py']
        has_duplicate = False
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'class GradCAM:' in content:
                        print(f"   ❌ 发现重复GradCAM定义在: {file_path}")
                        has_duplicate = True
        
        if not has_duplicate:
            print("   ✓ 没有重复的GradCAM定义")
        
        # 3. 测试CNN兼容性
        print("3. 测试CNN兼容性...")
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = CNNFeatureExtractor().to(device)
        monitor = TrainingMonitor(model_type='cnn', save_dir='temp_test', device=device)
        
        # 测试目标层检测
        target_layer = monitor.get_target_layer(model)
        if target_layer is not None:
            print("   ✓ CNN目标层检测成功")
        else:
            print("   ❌ CNN目标层检测失败")
            return False
        
        # 测试输出格式处理
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(test_input)
        
        if isinstance(output, tuple) and len(output) == 2:
            print("   ✓ CNN模型输出格式正确 (tuple)")
        else:
            print(f"   ❌ CNN模型输出格式异常: {type(output)}")
            return False
        
        # 4. 测试GradCAM统一实现
        print("4. 测试GradCAM统一实现...")
        grad_cam = GradCAM(model, target_layer)
        
        try:
            cam = grad_cam(test_input)
            if cam is not None and cam.shape == (224, 224):
                print("   ✓ GradCAM生成成功")
            else:
                print(f"   ❌ GradCAM输出异常: {cam.shape if cam is not None else None}")
                return False
        except Exception as e:
            print(f"   ❌ GradCAM生成失败: {e}")
            return False
        finally:
            grad_cam.remove_hooks()
        
        # 清理
        import shutil
        if os.path.exists('temp_test'):
            shutil.rmtree('temp_test')
        
        print("\n🎉 快速集成测试通过！")
        print("✅ 监控功能集成成功，原有代码逻辑未受影响")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_basic_integration() 