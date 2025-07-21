#!/usr/bin/env python3
"""
增强版批处理损失分析工具 - 统计多个最佳模型的训练损失与验证损失
支持云端和本地环境，增强了training.log解析和检查点损失提取功能
"""

import os
import json
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
import glob
import torch
import warnings
warnings.filterwarnings("ignore")

class EnhancedBatchLossAnalyzer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.getcwd()
        self.models_data = []
        self.setup_chinese_font()
    
    def setup_chinese_font(self):
        """设置中文字体"""
        try:
            import matplotlib.font_manager as fm
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            for font in chinese_fonts:
                if font in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"✓ 使用中文字体: {font}")
                    return
            print("⚠️ 未找到中文字体，使用英文标签")
        except Exception as e:
            print(f"字体设置失败: {e}")
    
    def detect_model_directories(self):
        """自动检测模型目录"""
        print("🔍 检测模型目录...")
        
        # 常见的模型目录模式
        patterns = [
            "*_results_*",
            "feature_training_*_results_*",
            "*_training_results_*",
            "model_*",
            "checkpoint_*"
        ]
        
        model_dirs = []
        for pattern in patterns:
            dirs = glob.glob(os.path.join(self.base_dir, pattern))
            model_dirs.extend([d for d in dirs if os.path.isdir(d)])
        
        # 去重并排序
        model_dirs = sorted(list(set(model_dirs)))
        
        print(f"📁 发现 {len(model_dirs)} 个模型目录:")
        for i, dir_path in enumerate(model_dirs, 1):
            dir_name = os.path.basename(dir_path)
            print(f"   {i:2d}. {dir_name}")
        
        return model_dirs
    
    def load_loss_data(self, model_dir):
        """从模型目录加载损失数据（增强版）"""
        model_name = os.path.basename(model_dir)
        print(f"\n📊 分析模型: {model_name}")
        
        loss_data = {
            'name': model_name,
            'path': model_dir,
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'best_epoch': None,
            'best_val_loss': None,
            'final_train_loss': None,
            'final_val_loss': None,
            'source_file': None
        }
        
        # 优先级顺序的损失数据文件
        loss_files = [
            'training_history.json',  # 最高优先级
            'training_log.json',
            'loss_history.json',
            'training_results.json',
            'losses.json',
            'log.json',
            'results.json',
            'history.json',
            'training_log.pkl',
            'loss_history.pkl',
            'training_history.pkl'
        ]
        
        # 首先尝试JSON和PKL文件
        for loss_file in loss_files:
            file_path = os.path.join(model_dir, loss_file)
            if os.path.exists(file_path):
                try:
                    data = self._load_file(file_path)
                    if self._extract_losses(data, loss_data):
                        loss_data['source_file'] = loss_file
                        print(f"   ✓ 从 {loss_file} 加载损失数据")
                        return self._finalize_loss_data(loss_data)
                except Exception as e:
                    print(f"   ⚠️ 加载 {loss_file} 失败: {e}")
                    continue
        
        # 尝试解析training.log作为纯文本
        training_log_path = os.path.join(model_dir, 'training.log')
        if os.path.exists(training_log_path):
            try:
                print(f"   📝 尝试解析training.log作为纯文本...")
                with open(training_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                
                if self._extract_losses_from_log(log_content, loss_data):
                    loss_data['source_file'] = 'training.log (parsed)'
                    print(f"   ✓ 从 training.log 解析损失数据")
                    return self._finalize_loss_data(loss_data)
            except Exception as e:
                print(f"   ⚠️ 解析 training.log 失败: {e}")
        
        # 最后尝试从checkpoint加载
        print(f"   📝 尝试从模型文件加载...")
        checkpoint_files = ['best_model.pth', 'checkpoint_latest.pth', 'checkpoint.pth', 'model.pth']
        for checkpoint_file in checkpoint_files:
            file_path = os.path.join(model_dir, checkpoint_file)
            if os.path.exists(file_path):
                try:
                    # 首先尝试安全加载
                    try:
                        checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
                    except Exception:
                        print(f"   📝 安全加载失败，尝试完整加载 {checkpoint_file}...")
                        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    
                    if self._extract_losses_from_checkpoint(checkpoint, loss_data):
                        loss_data['source_file'] = checkpoint_file
                        print(f"   ✓ 从 {checkpoint_file} 加载损失数据")
                        return self._finalize_loss_data(loss_data)
                except Exception as e:
                    print(f"   ⚠️ 加载 {checkpoint_file} 失败: {e}")
                    continue
        
        print(f"   ❌ 未找到损失数据")
        return None
    
    def _load_file(self, file_path):
        """加载文件（JSON或PKL）"""
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    
    def _extract_losses(self, data, loss_data):
        """从数据中提取损失值"""
        if isinstance(data, dict):
            # 标准格式: {'train_losses': [...], 'val_losses': [...]}
            if 'train_losses' in data and 'val_losses' in data:
                loss_data['train_losses'] = data['train_losses']
                loss_data['val_losses'] = data['val_losses']
                if 'epochs' in data:
                    loss_data['epochs'] = data['epochs']
                else:
                    loss_data['epochs'] = list(range(1, len(data['train_losses']) + 1))
                
                # 提取其他信息
                if 'best_epoch' in data:
                    loss_data['best_epoch'] = data['best_epoch']
                if 'best_val_loss' in data:
                    loss_data['best_val_loss'] = data['best_val_loss']
                
                return True
        
        return False
    
    def _extract_losses_from_checkpoint(self, checkpoint, loss_data):
        """从checkpoint中提取损失数据（增强版）"""
        print(f"   🔍 检查checkpoint内容，可用键: {list(checkpoint.keys())}")
        
        # 尝试完整的训练历史
        history_keys = [
            ('train_losses', 'val_losses'),
            ('train_loss_history', 'val_loss_history'),
            ('training_losses', 'validation_losses'),
            ('loss_history', 'val_loss_history')
        ]
        
        for train_key, val_key in history_keys:
            if train_key in checkpoint and val_key in checkpoint:
                train_losses = checkpoint[train_key]
                val_losses = checkpoint[val_key]
                
                if isinstance(train_losses, (list, tuple)) and isinstance(val_losses, (list, tuple)):
                    if len(train_losses) > 0 and len(val_losses) > 0:
                        loss_data['train_losses'] = list(train_losses)
                        loss_data['val_losses'] = list(val_losses)
                        loss_data['epochs'] = list(range(1, len(train_losses) + 1))
                        print(f"   ✓ 找到训练历史: {train_key}, {val_key}")
                        return True
        
        # 尝试从嵌套的history字典中提取
        if 'history' in checkpoint and isinstance(checkpoint['history'], dict):
            history = checkpoint['history']
            for train_key in ['loss', 'train_loss', 'training_loss', 'train_losses']:
                for val_key in ['val_loss', 'validation_loss', 'val_losses', 'valid_loss']:
                    if train_key in history and val_key in history:
                        train_losses = history[train_key]
                        val_losses = history[val_key]
                        
                        if isinstance(train_losses, (list, tuple)) and isinstance(val_losses, (list, tuple)):
                            if len(train_losses) > 0 and len(val_losses) > 0:
                                loss_data['train_losses'] = list(train_losses)
                                loss_data['val_losses'] = list(val_losses)
                                loss_data['epochs'] = list(range(1, len(train_losses) + 1))
                                print(f"   ✓ 从history中找到损失: {train_key}, {val_key}")
                                return True
        
        # 【新增】尝试从单独的验证损失中提取信息
        val_loss_keys = ['best_val_loss', 'val_loss', 'validation_loss', 'best_validation_loss']
        epoch_keys = ['best_epoch', 'epoch', 'best_epoch_num']
        
        best_val_loss = None
        best_epoch = None
        
        for val_key in val_loss_keys:
            if val_key in checkpoint:
                best_val_loss = float(checkpoint[val_key])
                break
        
        for epoch_key in epoch_keys:
            if epoch_key in checkpoint:
                best_epoch = int(checkpoint[epoch_key])
                break
        
        if best_val_loss is not None:
            loss_data['best_val_loss'] = best_val_loss
            if best_epoch is not None:
                loss_data['best_epoch'] = best_epoch
                print(f"   ✓ 找到最佳验证损失: {best_val_loss:.4f} (第{best_epoch}轮)")
            else:
                print(f"   ✓ 找到验证损失: {best_val_loss:.4f}")
            return True
        
        print(f"   ⚠️ 未在checkpoint中找到损失数据")
        return False
    
    def _extract_losses_from_log(self, log_content, loss_data):
        """从training.log文件中提取损失数据（新增功能）"""
        print(f"   🔍 尝试从training.log解析损失数据...")
        
        train_losses = []
        val_losses = []
        epochs = []
        
        lines = log_content.split('\n')
        current_epoch = None
        epoch_train_loss = None
        epoch_val_loss = None
        
        # 基线CNN格式：轮次 X 总结
        epoch_summary_pattern = r'轮次\s+(\d+)\s+总结:'
        train_loss_pattern = r'训练损失:\s*([\d.]+)'
        val_loss_pattern = r'验证损失:\s*([\d.]+)'
        
        # VGG格式：Epoch X 验证损失
        vgg_epoch_pattern = r'Epoch\s+(\d+)\s+验证损失:\s*([\d.]+)'
        
        # 增强CNN格式
        enhanced_epoch_pattern = r'--- 轮次\s+(\d+)/\d+ ---'
        enhanced_summary_pattern = r'第(\d+)轮总结:'
        
        for line in lines:
            line = line.strip()
            
            # 基线CNN格式解析
            epoch_match = re.search(epoch_summary_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            
            if current_epoch is not None:
                train_loss_match = re.search(train_loss_pattern, line)
                if train_loss_match:
                    epoch_train_loss = float(train_loss_match.group(1))
                
                val_loss_match = re.search(val_loss_pattern, line)
                if val_loss_match:
                    epoch_val_loss = float(val_loss_match.group(1))
                
                # 如果找到了该轮次的训练和验证损失，记录它们
                if epoch_train_loss is not None and epoch_val_loss is not None:
                    epochs.append(current_epoch)
                    train_losses.append(epoch_train_loss)
                    val_losses.append(epoch_val_loss)
                    
                    # 重置为下一轮次
                    current_epoch = None
                    epoch_train_loss = None
                    epoch_val_loss = None
            
            # VGG格式解析
            vgg_match = re.search(vgg_epoch_pattern, line)
            if vgg_match:
                epoch = int(vgg_match.group(1))
                val_loss = float(vgg_match.group(2))
                
                epochs.append(epoch)
                val_losses.append(val_loss)
                # VGG日志通常不记录训练损失，使用None占位
                train_losses.append(None)
        
        # 如果成功解析到数据
        if epochs and val_losses:
            loss_data['epochs'] = epochs
            loss_data['val_losses'] = val_losses
            
            # 过滤掉None值的训练损失
            valid_train_losses = [loss for loss in train_losses if loss is not None]
            if valid_train_losses and len(valid_train_losses) == len(val_losses):
                loss_data['train_losses'] = valid_train_losses
                print(f"   ✓ 从training.log解析到完整数据: {len(epochs)}轮，训练+验证损失")
            else:
                print(f"   ✓ 从training.log解析到验证损失: {len(epochs)}轮")
            
            return True
        
        print(f"   ❌ 无法从training.log解析损失数据")
        return False
    
    def _finalize_loss_data(self, loss_data):
        """完成损失数据的处理"""
        # 计算统计信息
        if loss_data['train_losses'] and loss_data['val_losses']:
            # 找到最佳验证损失
            val_losses = loss_data['val_losses']
            best_idx = np.argmin(val_losses)
            
            if loss_data['best_epoch'] is None:
                loss_data['best_epoch'] = loss_data['epochs'][best_idx] if loss_data['epochs'] else best_idx + 1
            if loss_data['best_val_loss'] is None:
                loss_data['best_val_loss'] = val_losses[best_idx]
            
            # 最终损失
            if loss_data['final_train_loss'] is None and loss_data['train_losses']:
                loss_data['final_train_loss'] = loss_data['train_losses'][-1]
            if loss_data['final_val_loss'] is None:
                loss_data['final_val_loss'] = loss_data['val_losses'][-1]
            
            print(f"   📈 训练轮数: {len(loss_data['val_losses'])}")
            print(f"   🎯 最佳验证损失: {loss_data['best_val_loss']:.4f} (第{loss_data['best_epoch']}轮)")
            if loss_data['final_train_loss'] is not None:
                print(f"   📊 最终训练损失: {loss_data['final_train_loss']:.4f}")
            print(f"   📊 最终验证损失: {loss_data['final_val_loss']:.4f}")
        elif loss_data['best_val_loss'] is not None:
            print(f"   📊 验证损失: {loss_data['best_val_loss']:.4f}")
            if loss_data['best_epoch'] is not None:
                print(f"   📊 最佳轮次: {loss_data['best_epoch']}")
        
        return loss_data
    
    def analyze_models(self, model_dirs=None):
        """分析多个模型的损失"""
        if model_dirs is None:
            model_dirs = self.detect_model_directories()
        
        if not model_dirs:
            print("❌ 未找到模型目录")
            return []
        
        print(f"\n🔍 开始分析 {len(model_dirs)} 个模型...")
        
        for model_dir in model_dirs:
            loss_data = self.load_loss_data(model_dir)
            if loss_data:
                self.models_data.append(loss_data)
        
        print(f"\n✅ 成功分析 {len(self.models_data)} 个模型")
        return self.models_data
    
    def create_summary_table(self, output_dir="loss_analysis"):
        """创建汇总表格"""
        if not self.models_data:
            print("❌ 没有模型数据可生成表格")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        table_data = []
        for model in self.models_data:
            row = {
                '模型名称': model['name'],
                '数据源': model['source_file'],
                '最佳验证损失': model['best_val_loss'] if model['best_val_loss'] is not None else 'N/A',
                '最佳轮次': model['best_epoch'] if model['best_epoch'] is not None else 'N/A',
                '最终验证损失': model['final_val_loss'] if model['final_val_loss'] is not None else 'N/A',
                '训练轮数': len(model['val_losses']) if model['val_losses'] else 'N/A'
            }
            table_data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(table_data)
        
        # 按最佳验证损失排序
        def sort_key(x):
            if x == 'N/A':
                return float('inf')
            return float(x)
        
        df_sorted = df.copy()
        df_sorted['sort_key'] = df_sorted['最佳验证损失'].apply(sort_key)
        df_sorted = df_sorted.sort_values('sort_key').drop('sort_key', axis=1)
        
        # 保存CSV
        csv_path = os.path.join(output_dir, 'model_summary.csv')
        df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存HTML表格
        html_path = os.path.join(output_dir, 'model_summary.html')
        df_sorted.to_html(html_path, index=False, escape=False, 
                         table_id='model_summary', classes='table table-striped')
        
        print(f"✅ 汇总表格已保存:")
        print(f"   CSV: {csv_path}")
        print(f"   HTML: {html_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版批处理损失分析工具')
    parser.add_argument('--base_dir', type=str, default=None, help='基础目录路径')
    parser.add_argument('--output_dir', type=str, default='enhanced_loss_analysis', help='输出目录')
    args = parser.parse_args()
    
    print("🚀 增强版批处理损失分析工具")
    print("=" * 60)
    print("新增功能:")
    print("- 支持从training.log解析VGG等训练日志")
    print("- 从检查点文件提取单独的验证损失信息")
    print("- 优化的模型目录检测和去重")
    print("- 增强的错误处理和兼容性")
    print("=" * 60)
    
    # 创建分析器
    analyzer = EnhancedBatchLossAnalyzer(args.base_dir)
    
    # 分析模型
    models_data = analyzer.analyze_models()
    
    if models_data:
        print(f"\n📈 生成分析结果...")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 生成汇总表格
        analyzer.create_summary_table(args.output_dir)
        
        print(f"\n✅ 分析完成！结果保存在: {args.output_dir}")
        print(f"📊 成功分析 {len(models_data)} 个模型")
        
        # 显示结果统计
        models_with_full_history = sum(1 for m in models_data if m['train_losses'] and m['val_losses'])
        models_with_val_only = sum(1 for m in models_data if m['best_val_loss'] is not None and not m['train_losses'])
        
        print(f"   - 完整训练历史: {models_with_full_history} 个")
        print(f"   - 仅验证损失: {models_with_val_only} 个")
    else:
        print(f"❌ 未找到可分析的模型数据")

if __name__ == "__main__":
    main() 
"""
增强版批处理损失分析工具 - 统计多个最佳模型的训练损失与验证损失
支持云端和本地环境，增强了training.log解析和检查点损失提取功能
"""

import os
import json
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
import glob
import torch
import warnings
warnings.filterwarnings("ignore")

class EnhancedBatchLossAnalyzer:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.getcwd()
        self.models_data = []
        self.setup_chinese_font()
    
    def setup_chinese_font(self):
        """设置中文字体"""
        try:
            import matplotlib.font_manager as fm
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            for font in chinese_fonts:
                if font in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"✓ 使用中文字体: {font}")
                    return
            print("⚠️ 未找到中文字体，使用英文标签")
        except Exception as e:
            print(f"字体设置失败: {e}")
    
    def detect_model_directories(self):
        """自动检测模型目录"""
        print("🔍 检测模型目录...")
        
        # 常见的模型目录模式
        patterns = [
            "*_results_*",
            "feature_training_*_results_*",
            "*_training_results_*",
            "model_*",
            "checkpoint_*"
        ]
        
        model_dirs = []
        for pattern in patterns:
            dirs = glob.glob(os.path.join(self.base_dir, pattern))
            model_dirs.extend([d for d in dirs if os.path.isdir(d)])
        
        # 去重并排序
        model_dirs = sorted(list(set(model_dirs)))
        
        print(f"📁 发现 {len(model_dirs)} 个模型目录:")
        for i, dir_path in enumerate(model_dirs, 1):
            dir_name = os.path.basename(dir_path)
            print(f"   {i:2d}. {dir_name}")
        
        return model_dirs
    
    def load_loss_data(self, model_dir):
        """从模型目录加载损失数据（增强版）"""
        model_name = os.path.basename(model_dir)
        print(f"\n📊 分析模型: {model_name}")
        
        loss_data = {
            'name': model_name,
            'path': model_dir,
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'best_epoch': None,
            'best_val_loss': None,
            'final_train_loss': None,
            'final_val_loss': None,
            'source_file': None
        }
        
        # 优先级顺序的损失数据文件
        loss_files = [
            'training_history.json',  # 最高优先级
            'training_log.json',
            'loss_history.json',
            'training_results.json',
            'losses.json',
            'log.json',
            'results.json',
            'history.json',
            'training_log.pkl',
            'loss_history.pkl',
            'training_history.pkl'
        ]
        
        # 首先尝试JSON和PKL文件
        for loss_file in loss_files:
            file_path = os.path.join(model_dir, loss_file)
            if os.path.exists(file_path):
                try:
                    data = self._load_file(file_path)
                    if self._extract_losses(data, loss_data):
                        loss_data['source_file'] = loss_file
                        print(f"   ✓ 从 {loss_file} 加载损失数据")
                        return self._finalize_loss_data(loss_data)
                except Exception as e:
                    print(f"   ⚠️ 加载 {loss_file} 失败: {e}")
                    continue
        
        # 尝试解析training.log作为纯文本
        training_log_path = os.path.join(model_dir, 'training.log')
        if os.path.exists(training_log_path):
            try:
                print(f"   📝 尝试解析training.log作为纯文本...")
                with open(training_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()
                
                if self._extract_losses_from_log(log_content, loss_data):
                    loss_data['source_file'] = 'training.log (parsed)'
                    print(f"   ✓ 从 training.log 解析损失数据")
                    return self._finalize_loss_data(loss_data)
            except Exception as e:
                print(f"   ⚠️ 解析 training.log 失败: {e}")
        
        # 最后尝试从checkpoint加载
        print(f"   📝 尝试从模型文件加载...")
        checkpoint_files = ['best_model.pth', 'checkpoint_latest.pth', 'checkpoint.pth', 'model.pth']
        for checkpoint_file in checkpoint_files:
            file_path = os.path.join(model_dir, checkpoint_file)
            if os.path.exists(file_path):
                try:
                    # 首先尝试安全加载
                    try:
                        checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
                    except Exception:
                        print(f"   📝 安全加载失败，尝试完整加载 {checkpoint_file}...")
                        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    
                    if self._extract_losses_from_checkpoint(checkpoint, loss_data):
                        loss_data['source_file'] = checkpoint_file
                        print(f"   ✓ 从 {checkpoint_file} 加载损失数据")
                        return self._finalize_loss_data(loss_data)
                except Exception as e:
                    print(f"   ⚠️ 加载 {checkpoint_file} 失败: {e}")
                    continue
        
        print(f"   ❌ 未找到损失数据")
        return None
    
    def _load_file(self, file_path):
        """加载文件（JSON或PKL）"""
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    
    def _extract_losses(self, data, loss_data):
        """从数据中提取损失值"""
        if isinstance(data, dict):
            # 标准格式: {'train_losses': [...], 'val_losses': [...]}
            if 'train_losses' in data and 'val_losses' in data:
                loss_data['train_losses'] = data['train_losses']
                loss_data['val_losses'] = data['val_losses']
                if 'epochs' in data:
                    loss_data['epochs'] = data['epochs']
                else:
                    loss_data['epochs'] = list(range(1, len(data['train_losses']) + 1))
                
                # 提取其他信息
                if 'best_epoch' in data:
                    loss_data['best_epoch'] = data['best_epoch']
                if 'best_val_loss' in data:
                    loss_data['best_val_loss'] = data['best_val_loss']
                
                return True
        
        return False
    
    def _extract_losses_from_checkpoint(self, checkpoint, loss_data):
        """从checkpoint中提取损失数据（增强版）"""
        print(f"   🔍 检查checkpoint内容，可用键: {list(checkpoint.keys())}")
        
        # 尝试完整的训练历史
        history_keys = [
            ('train_losses', 'val_losses'),
            ('train_loss_history', 'val_loss_history'),
            ('training_losses', 'validation_losses'),
            ('loss_history', 'val_loss_history')
        ]
        
        for train_key, val_key in history_keys:
            if train_key in checkpoint and val_key in checkpoint:
                train_losses = checkpoint[train_key]
                val_losses = checkpoint[val_key]
                
                if isinstance(train_losses, (list, tuple)) and isinstance(val_losses, (list, tuple)):
                    if len(train_losses) > 0 and len(val_losses) > 0:
                        loss_data['train_losses'] = list(train_losses)
                        loss_data['val_losses'] = list(val_losses)
                        loss_data['epochs'] = list(range(1, len(train_losses) + 1))
                        print(f"   ✓ 找到训练历史: {train_key}, {val_key}")
                        return True
        
        # 尝试从嵌套的history字典中提取
        if 'history' in checkpoint and isinstance(checkpoint['history'], dict):
            history = checkpoint['history']
            for train_key in ['loss', 'train_loss', 'training_loss', 'train_losses']:
                for val_key in ['val_loss', 'validation_loss', 'val_losses', 'valid_loss']:
                    if train_key in history and val_key in history:
                        train_losses = history[train_key]
                        val_losses = history[val_key]
                        
                        if isinstance(train_losses, (list, tuple)) and isinstance(val_losses, (list, tuple)):
                            if len(train_losses) > 0 and len(val_losses) > 0:
                                loss_data['train_losses'] = list(train_losses)
                                loss_data['val_losses'] = list(val_losses)
                                loss_data['epochs'] = list(range(1, len(train_losses) + 1))
                                print(f"   ✓ 从history中找到损失: {train_key}, {val_key}")
                                return True
        
        # 【新增】尝试从单独的验证损失中提取信息
        val_loss_keys = ['best_val_loss', 'val_loss', 'validation_loss', 'best_validation_loss']
        epoch_keys = ['best_epoch', 'epoch', 'best_epoch_num']
        
        best_val_loss = None
        best_epoch = None
        
        for val_key in val_loss_keys:
            if val_key in checkpoint:
                best_val_loss = float(checkpoint[val_key])
                break
        
        for epoch_key in epoch_keys:
            if epoch_key in checkpoint:
                best_epoch = int(checkpoint[epoch_key])
                break
        
        if best_val_loss is not None:
            loss_data['best_val_loss'] = best_val_loss
            if best_epoch is not None:
                loss_data['best_epoch'] = best_epoch
                print(f"   ✓ 找到最佳验证损失: {best_val_loss:.4f} (第{best_epoch}轮)")
            else:
                print(f"   ✓ 找到验证损失: {best_val_loss:.4f}")
            return True
        
        print(f"   ⚠️ 未在checkpoint中找到损失数据")
        return False
    
    def _extract_losses_from_log(self, log_content, loss_data):
        """从training.log文件中提取损失数据（新增功能）"""
        print(f"   🔍 尝试从training.log解析损失数据...")
        
        train_losses = []
        val_losses = []
        epochs = []
        
        lines = log_content.split('\n')
        current_epoch = None
        epoch_train_loss = None
        epoch_val_loss = None
        
        # 基线CNN格式：轮次 X 总结
        epoch_summary_pattern = r'轮次\s+(\d+)\s+总结:'
        train_loss_pattern = r'训练损失:\s*([\d.]+)'
        val_loss_pattern = r'验证损失:\s*([\d.]+)'
        
        # VGG格式：Epoch X 验证损失
        vgg_epoch_pattern = r'Epoch\s+(\d+)\s+验证损失:\s*([\d.]+)'
        
        # 增强CNN格式
        enhanced_epoch_pattern = r'--- 轮次\s+(\d+)/\d+ ---'
        enhanced_summary_pattern = r'第(\d+)轮总结:'
        
        for line in lines:
            line = line.strip()
            
            # 基线CNN格式解析
            epoch_match = re.search(epoch_summary_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            
            if current_epoch is not None:
                train_loss_match = re.search(train_loss_pattern, line)
                if train_loss_match:
                    epoch_train_loss = float(train_loss_match.group(1))
                
                val_loss_match = re.search(val_loss_pattern, line)
                if val_loss_match:
                    epoch_val_loss = float(val_loss_match.group(1))
                
                # 如果找到了该轮次的训练和验证损失，记录它们
                if epoch_train_loss is not None and epoch_val_loss is not None:
                    epochs.append(current_epoch)
                    train_losses.append(epoch_train_loss)
                    val_losses.append(epoch_val_loss)
                    
                    # 重置为下一轮次
                    current_epoch = None
                    epoch_train_loss = None
                    epoch_val_loss = None
            
            # VGG格式解析
            vgg_match = re.search(vgg_epoch_pattern, line)
            if vgg_match:
                epoch = int(vgg_match.group(1))
                val_loss = float(vgg_match.group(2))
                
                epochs.append(epoch)
                val_losses.append(val_loss)
                # VGG日志通常不记录训练损失，使用None占位
                train_losses.append(None)
        
        # 如果成功解析到数据
        if epochs and val_losses:
            loss_data['epochs'] = epochs
            loss_data['val_losses'] = val_losses
            
            # 过滤掉None值的训练损失
            valid_train_losses = [loss for loss in train_losses if loss is not None]
            if valid_train_losses and len(valid_train_losses) == len(val_losses):
                loss_data['train_losses'] = valid_train_losses
                print(f"   ✓ 从training.log解析到完整数据: {len(epochs)}轮，训练+验证损失")
            else:
                print(f"   ✓ 从training.log解析到验证损失: {len(epochs)}轮")
            
            return True
        
        print(f"   ❌ 无法从training.log解析损失数据")
        return False
    
    def _finalize_loss_data(self, loss_data):
        """完成损失数据的处理"""
        # 计算统计信息
        if loss_data['train_losses'] and loss_data['val_losses']:
            # 找到最佳验证损失
            val_losses = loss_data['val_losses']
            best_idx = np.argmin(val_losses)
            
            if loss_data['best_epoch'] is None:
                loss_data['best_epoch'] = loss_data['epochs'][best_idx] if loss_data['epochs'] else best_idx + 1
            if loss_data['best_val_loss'] is None:
                loss_data['best_val_loss'] = val_losses[best_idx]
            
            # 最终损失
            if loss_data['final_train_loss'] is None and loss_data['train_losses']:
                loss_data['final_train_loss'] = loss_data['train_losses'][-1]
            if loss_data['final_val_loss'] is None:
                loss_data['final_val_loss'] = loss_data['val_losses'][-1]
            
            print(f"   📈 训练轮数: {len(loss_data['val_losses'])}")
            print(f"   🎯 最佳验证损失: {loss_data['best_val_loss']:.4f} (第{loss_data['best_epoch']}轮)")
            if loss_data['final_train_loss'] is not None:
                print(f"   📊 最终训练损失: {loss_data['final_train_loss']:.4f}")
            print(f"   📊 最终验证损失: {loss_data['final_val_loss']:.4f}")
        elif loss_data['best_val_loss'] is not None:
            print(f"   📊 验证损失: {loss_data['best_val_loss']:.4f}")
            if loss_data['best_epoch'] is not None:
                print(f"   📊 最佳轮次: {loss_data['best_epoch']}")
        
        return loss_data
    
    def analyze_models(self, model_dirs=None):
        """分析多个模型的损失"""
        if model_dirs is None:
            model_dirs = self.detect_model_directories()
        
        if not model_dirs:
            print("❌ 未找到模型目录")
            return []
        
        print(f"\n🔍 开始分析 {len(model_dirs)} 个模型...")
        
        for model_dir in model_dirs:
            loss_data = self.load_loss_data(model_dir)
            if loss_data:
                self.models_data.append(loss_data)
        
        print(f"\n✅ 成功分析 {len(self.models_data)} 个模型")
        return self.models_data
    
    def create_summary_table(self, output_dir="loss_analysis"):
        """创建汇总表格"""
        if not self.models_data:
            print("❌ 没有模型数据可生成表格")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        table_data = []
        for model in self.models_data:
            row = {
                '模型名称': model['name'],
                '数据源': model['source_file'],
                '最佳验证损失': model['best_val_loss'] if model['best_val_loss'] is not None else 'N/A',
                '最佳轮次': model['best_epoch'] if model['best_epoch'] is not None else 'N/A',
                '最终验证损失': model['final_val_loss'] if model['final_val_loss'] is not None else 'N/A',
                '训练轮数': len(model['val_losses']) if model['val_losses'] else 'N/A'
            }
            table_data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(table_data)
        
        # 按最佳验证损失排序
        def sort_key(x):
            if x == 'N/A':
                return float('inf')
            return float(x)
        
        df_sorted = df.copy()
        df_sorted['sort_key'] = df_sorted['最佳验证损失'].apply(sort_key)
        df_sorted = df_sorted.sort_values('sort_key').drop('sort_key', axis=1)
        
        # 保存CSV
        csv_path = os.path.join(output_dir, 'model_summary.csv')
        df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存HTML表格
        html_path = os.path.join(output_dir, 'model_summary.html')
        df_sorted.to_html(html_path, index=False, escape=False, 
                         table_id='model_summary', classes='table table-striped')
        
        print(f"✅ 汇总表格已保存:")
        print(f"   CSV: {csv_path}")
        print(f"   HTML: {html_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强版批处理损失分析工具')
    parser.add_argument('--base_dir', type=str, default=None, help='基础目录路径')
    parser.add_argument('--output_dir', type=str, default='enhanced_loss_analysis', help='输出目录')
    args = parser.parse_args()
    
    print("🚀 增强版批处理损失分析工具")
    print("=" * 60)
    print("新增功能:")
    print("- 支持从training.log解析VGG等训练日志")
    print("- 从检查点文件提取单独的验证损失信息")
    print("- 优化的模型目录检测和去重")
    print("- 增强的错误处理和兼容性")
    print("=" * 60)
    
    # 创建分析器
    analyzer = EnhancedBatchLossAnalyzer(args.base_dir)
    
    # 分析模型
    models_data = analyzer.analyze_models()
    
    if models_data:
        print(f"\n📈 生成分析结果...")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 生成汇总表格
        analyzer.create_summary_table(args.output_dir)
        
        print(f"\n✅ 分析完成！结果保存在: {args.output_dir}")
        print(f"📊 成功分析 {len(models_data)} 个模型")
        
        # 显示结果统计
        models_with_full_history = sum(1 for m in models_data if m['train_losses'] and m['val_losses'])
        models_with_val_only = sum(1 for m in models_data if m['best_val_loss'] is not None and not m['train_losses'])
        
        print(f"   - 完整训练历史: {models_with_full_history} 个")
        print(f"   - 仅验证损失: {models_with_val_only} 个")
    else:
        print(f"❌ 未找到可分析的模型数据")

if __name__ == "__main__":
    main() 
 
 
 