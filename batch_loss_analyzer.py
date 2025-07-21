#!/usr/bin/env python3
"""
批处理损失分析工具 - 统计多个最佳模型的训练损失与验证损失
支持云端和本地环境
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

class BatchLossAnalyzer:
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
            "feature_training_*_results_*",
            "*_training_results_*",
            "*_results_*",
            "model_*",
            "checkpoint_*"
        ]
        
        model_dirs = []
        for pattern in patterns:
            dirs = glob.glob(os.path.join(self.base_dir, pattern))
            model_dirs.extend([d for d in dirs if os.path.isdir(d)])
        
        # 去重并排序
        model_dirs = sorted(list(set(model_dirs)))
        
        # 去重：对于前缀相同的目录，只保留最新的
        model_dirs = self._deduplicate_model_directories(model_dirs)
        
        print(f"📁 发现 {len(model_dirs)} 个模型目录 (已去重):")
        for i, dir_path in enumerate(model_dirs, 1):
            dir_name = os.path.basename(dir_path)
            print(f"   {i:2d}. {dir_name}")
        
        return model_dirs
    
    def _deduplicate_model_directories(self, model_dirs):
        """去重模型目录，对于前缀相同的目录只保留最佳验证损失最小的目录"""
        if not model_dirs:
            return model_dirs
        
        # 按前缀分组
        prefix_groups = {}
        
        for dir_path in model_dirs:
            dir_name = os.path.basename(dir_path)
            # 提取前缀（去除时间戳部分）
            prefix = self._extract_model_prefix(dir_name)
            
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(dir_path)
        
        # 对每个前缀组，只保留最佳验证损失最小的目录
        deduplicated = []
        removed_count = 0
        
        for prefix, dirs in prefix_groups.items():
            if len(dirs) > 1:
                print(f"   🔍 前缀 '{prefix}' 有 {len(dirs)} 个版本，正在比较验证损失...")
                
                # 获取每个目录的验证损失
                dirs_with_loss = []
                for dir_path in dirs:
                    dir_name = os.path.basename(dir_path)
                    timestamp = self._extract_timestamp(dir_name)
                    
                    # 尝试加载损失数据
                    best_val_loss = self._get_best_val_loss(dir_path)
                    
                    dirs_with_loss.append((best_val_loss, timestamp, dir_path, dir_name))
                    
                    if best_val_loss is not None:
                        print(f"      📊 {dir_name}: 最佳验证损失 = {best_val_loss:.4f}")
                    else:
                        print(f"      ❌ {dir_name}: 无法获取验证损失")
                
                # 排序策略：
                # 1. 优先选择有验证损失数据的目录
                # 2. 在有验证损失的目录中选择损失最小的
                # 3. 如果损失相同，选择时间戳最新的
                # 4. 如果都没有损失数据，选择时间戳最新的
                def sort_key(item):
                    best_val_loss, timestamp, dir_path, dir_name = item
                    if best_val_loss is not None:
                        return (0, best_val_loss, -float(timestamp.replace('_', '')))  # 损失小的优先，时间新的优先
                    else:
                        return (1, 0, -float(timestamp.replace('_', '')))  # 没有损失数据的排后面
                
                dirs_with_loss.sort(key=sort_key)
                
                # 保留排序后的第一个（最优的）
                selected_loss, selected_timestamp, selected_dir, selected_name = dirs_with_loss[0]
                deduplicated.append(selected_dir)
                
                # 记录被移除的目录
                removed_dirs = dirs_with_loss[1:]
                removed_count += len(removed_dirs)
                
                if selected_loss is not None:
                    print(f"   🏆 保留最佳模型: {selected_name} (验证损失: {selected_loss:.4f})")
                else:
                    print(f"   📅 保留最新模型: {selected_name} (无损失数据，按时间戳选择)")
                
                for removed_loss, removed_timestamp, removed_dir, removed_name in removed_dirs:
                    if removed_loss is not None:
                        print(f"      ⏭️ 跳过: {removed_name} (验证损失: {removed_loss:.4f})")
                    else:
                        print(f"      ⏭️ 跳过: {removed_name} (无损失数据)")
            else:
                deduplicated.append(dirs[0])
        
        if removed_count > 0:
            print(f"   📊 去重统计: 原有 {len(model_dirs)} 个目录，去重后 {len(deduplicated)} 个目录")
        
        return deduplicated
    
    def _get_best_val_loss(self, model_dir):
        """获取模型目录的最佳验证损失"""
        try:
            # 临时创建一个损失数据字典
            temp_loss_data = {
                'name': os.path.basename(model_dir),
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
            
            # 尝试多种损失数据文件（JSON和PKL）
            loss_files = [
                'training_log.json',
                'training.log',
                'loss_history.json',
                'training_history.json',
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
                        if self._extract_losses(data, temp_loss_data):
                            temp_loss_data['source_file'] = loss_file
                            break
                    except Exception:
                        continue
            
            # 如果JSON/PKL文件都失败了，尝试从checkpoint加载
            if not temp_loss_data['source_file']:
                checkpoint_files = ['best_model.pth', 'checkpoint_latest.pth', 'checkpoint.pth', 'model.pth']
                for checkpoint_file in checkpoint_files:
                    file_path = os.path.join(model_dir, checkpoint_file)
                    if os.path.exists(file_path):
                        try:
                            # 首先尝试安全加载（weights_only=True）
                            try:
                                checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
                            except Exception:
                                # 如果安全加载失败，尝试完整加载
                                checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                            
                            if self._extract_losses_from_checkpoint(checkpoint, temp_loss_data):
                                temp_loss_data['source_file'] = checkpoint_file
                                break
                        except Exception:
                            continue
            
            # 计算最佳验证损失
            if temp_loss_data['val_losses']:
                best_val_loss = min(temp_loss_data['val_losses'])
                return best_val_loss
            elif temp_loss_data['final_val_loss'] is not None:
                return temp_loss_data['final_val_loss']
            else:
                return None
                
        except Exception:
            return None
    
    def _extract_model_prefix(self, dir_name):
        """提取模型目录的前缀（去除时间戳）"""
        # 移除常见的时间戳格式
        patterns = [
            r'_\d{8}_\d{6}.*$',    # _20241201_123456
            r'_\d{8}.*$',          # _20241201
            r'_results_\d{8}_\d{6}.*$',  # _results_20241201_123456
            r'_results_\d{8}.*$'   # _results_20241201
        ]
        
        prefix = dir_name
        for pattern in patterns:
            prefix = re.sub(pattern, '', prefix)
        
        return prefix
    
    def _extract_timestamp(self, dir_name):
        """提取目录名中的时间戳用于排序"""
        # 尝试提取不同格式的时间戳
        timestamp_patterns = [
            r'(\d{8}_\d{6})',      # 20241201_123456
            r'(\d{8})',            # 20241201
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, dir_name)
            if match:
                timestamp_str = match.group(1)
                # 标准化为可比较的格式
                if '_' in timestamp_str:
                    return timestamp_str  # 20241201_123456
                else:
                    return timestamp_str + '_000000'  # 20241201 -> 20241201_000000
        
        # 如果没找到时间戳，返回目录名本身作为排序依据
        return dir_name
    
    def _extract_model_type(self, model_name):
        """提取模型类型"""
        name = model_name.lower()
        
        # 本地模型类型识别（完整匹配）
        local_models = [
            'baseline_cnn', 'enhanced_cnn_resnet50', 'enhanced_cnn_vgg16', 
            'cnn_vgg', 'cnn_resnet50', 'cnn_resnet', 'cnn_vgg16'
        ]
        
        for model in local_models:
            if model in name:
                # 简化名称
                if 'baseline_cnn' in model:
                    return 'baseline_cnn'
                elif 'enhanced_cnn_resnet50' in model:
                    return 'cnn_resnet50'
                elif 'enhanced_cnn_vgg16' in model:
                    return 'cnn_vgg16'
                elif 'cnn_vgg' in model:
                    return 'cnn_vgg'
                elif 'cnn_resnet50' in model:
                    return 'cnn_resnet50'
                else:
                    return model
        
        # 云端模型类型识别
        cloud_models = [
            'vgg_unfrozen', 'vgg', 'resnet50_unfrozen', 'resnet50', 
            'resnet', 'densenet', 'efficientnet', 'mobilenet'
        ]
        
        for model in cloud_models:
            if model in name:
                return model
        
        # 通用模型类型识别
        general_models = [
            'transformer', 'bert', 'gpt', 'lstm', 'rnn', 'alexnet', 
            'inception', 'squeezenet', 'shufflenet'
        ]
        
        for model in general_models:
            if model in name:
                return model
        
        # 如果都没找到，尝试提取第一个有意义的词
        parts = model_name.split('_')
        for part in parts:
            if len(part) > 2 and part.isalpha():
                return part.lower()
        
        return 'model'
    
    def _extract_training_mode(self, model_name):
        """提取训练模式"""
        name = model_name.lower()
        
        # 训练模式模式匹配
        training_modes = [
            r'bg0[-_](\d+)mw',      # bg0-20mw, bg0_100mw
            r'bg1[-_](\d+)mw',      # bg1-100mw, bg1_400mw
            r'bg(\d+)[-_](\d+)mw',  # bg0-20mw, bg1-100mw
            r'bg(\d+)',             # bg0, bg1
            r'all',                 # all
            r'frozen',              # frozen
            r'unfrozen',            # unfrozen
            r'finetune',            # finetune
            r'pretrained',          # pretrained
            r'scratch'              # from scratch
        ]
        
        for pattern in training_modes:
            match = re.search(pattern, name)
            if match:
                if 'bg0' in match.group(0) and 'mw' in match.group(0):
                    # 提取bg0-XXmw格式
                    mw_match = re.search(r'bg0[-_](\d+)mw', match.group(0))
                    if mw_match:
                        return f"bg0_{mw_match.group(1)}mw"
                elif 'bg1' in match.group(0) and 'mw' in match.group(0):
                    # 提取bg1-XXmw格式
                    mw_match = re.search(r'bg1[-_](\d+)mw', match.group(0))
                    if mw_match:
                        return f"bg1_{mw_match.group(1)}mw"
                elif 'bg' in match.group(0):
                    # 提取bgX格式
                    bg_match = re.search(r'bg(\d+)', match.group(0))
                    if bg_match:
                        return f"bg{bg_match.group(1)}"
                else:
                    return match.group(0)
        
        # 如果没有找到标准训练模式，尝试提取其他有意义的标识符
        parts = model_name.split('_')
        meaningful_parts = []
        
        for part in parts:
            part_lower = part.lower()
            # 跳过时间戳、results等无关部分
            if (not re.match(r'\d{8}', part) and 
                not re.match(r'\d{6}', part) and
                part_lower not in ['results', 'training', 'feature', 'enhanced', 'baseline']):
                if len(part) > 1 and not part.isdigit():
                    meaningful_parts.append(part_lower)
        
        # 返回最后一个有意义的部分作为训练模式
        if meaningful_parts:
            # 优先返回包含数字或特殊标识的部分
            for part in meaningful_parts:
                if re.search(r'\d', part) or part in ['all', 'frozen', 'unfrozen']:
                    return part
            return meaningful_parts[-1]
        
        return None
    
    def load_loss_data(self, model_dir):
        """从模型目录加载损失数据"""
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
        
        # 尝试多种损失数据文件（JSON和PKL）
        loss_files = [
            'training_log.json',
            'training.log',  # 添加支持
            'loss_history.json',
            'training_history.json',
            'training_results.json',
            'losses.json',
            'log.json',
            'results.json',  # 添加支持
            'history.json',  # 添加支持
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
                        break
                except Exception as e:
                    print(f"   ⚠️ 加载 {loss_file} 失败: {e}")
                    continue  # 继续尝试下一个文件
        
        # 如果JSON/PKL文件都失败了，尝试从checkpoint加载
        if not loss_data['source_file']:
            print(f"   📝 JSON/PKL文件加载失败，尝试从模型文件加载...")
            checkpoint_files = ['best_model.pth', 'checkpoint_latest.pth', 'checkpoint.pth', 'model.pth']
            for checkpoint_file in checkpoint_files:
                file_path = os.path.join(model_dir, checkpoint_file)
                if os.path.exists(file_path):
                    try:
                        # 首先尝试安全加载（weights_only=True）
                        try:
                            checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
                        except Exception as safe_load_error:
                            # 如果安全加载失败，尝试完整加载（适用于包含argparse.Namespace等对象的文件）
                            print(f"   📝 安全加载失败，尝试完整加载 {checkpoint_file}...")
                            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                        
                        if self._extract_losses_from_checkpoint(checkpoint, loss_data):
                            loss_data['source_file'] = checkpoint_file
                            print(f"   ✓ 从 {checkpoint_file} 加载损失数据")
                            break
                    except Exception as e:
                        print(f"   ⚠️ 加载 {checkpoint_file} 失败: {e}")
                        continue  # 继续尝试下一个文件
        
        if loss_data['source_file']:
            self._calculate_statistics(loss_data)
            return loss_data
        else:
            print(f"   ❌ 未找到损失数据")
            return None
    
    def _load_file(self, file_path):
        """加载文件（JSON或PKL）"""
        file_name = os.path.basename(file_path)
        
        if file_path.endswith('.json') or file_path.endswith('.log'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # 如果文件为空，跳过
                if not content:
                    raise ValueError("文件为空")
                
                # 尝试直接解析JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ JSON解析失败 ({file_name}): {e}")
                    
                    # 尝试修复常见的JSON问题
                    # 1. 移除尾随逗号
                    content = content.rstrip().rstrip(',')
                    
                    # 2. 尝试逐行解析（处理多个JSON对象的情况）
                    lines = content.split('\n')
                    json_objects = []
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):  # 跳过注释行
                            try:
                                obj = json.loads(line)
                                json_objects.append(obj)
                            except:
                                continue
                    
                    if json_objects:
                        print(f"   📝 找到 {len(json_objects)} 个JSON对象")
                        
                        # 如果有多个JSON对象，尝试构建训练历史
                        if len(json_objects) > 1:
                            # 尝试从多个epoch记录构建历史
                            train_losses = []
                            val_losses = []
                            epochs = []
                            
                            for obj in json_objects:
                                if isinstance(obj, dict):
                                    # 尝试不同的键名组合
                                    train_loss_keys = ['train_loss', 'training_loss', 'loss']
                                    val_loss_keys = ['val_loss', 'validation_loss', 'valid_loss']
                                    epoch_keys = ['epoch', 'step', 'iteration']
                                    
                                    train_loss = None
                                    val_loss = None
                                    epoch = None
                                    
                                    for key in train_loss_keys:
                                        if key in obj:
                                            train_loss = float(obj[key])
                                            break
                                    
                                    for key in val_loss_keys:
                                        if key in obj:
                                            val_loss = float(obj[key])
                                            break
                                    
                                    for key in epoch_keys:
                                        if key in obj:
                                            epoch = int(obj[key])
                                            break
                                    
                                    if train_loss is not None and val_loss is not None:
                                        train_losses.append(train_loss)
                                        val_losses.append(val_loss)
                                        epochs.append(epoch if epoch is not None else len(epochs) + 1)
                            
                            if train_losses and val_losses:
                                return {
                                    'train_losses': train_losses,
                                    'val_losses': val_losses,
                                    'epochs': epochs
                                }
                        
                        # 单个对象或合并策略
                        if len(json_objects) == 1:
                            return json_objects[0]
                        else:
                            # 尝试合并多个对象
                            merged = {}
                            for obj in json_objects:
                                if isinstance(obj, dict):
                                    merged.update(obj)
                            return merged if merged else json_objects
                    
                    # 如果还是失败，尝试提取部分有效的JSON
                    # 查找可能的JSON片段
                    import re
                    json_pattern = r'\{[^{}]*\}'
                    matches = re.findall(json_pattern, content)
                    
                    for match in matches:
                        try:
                            return json.loads(match)
                        except:
                            continue
                    
                    # 最后尝试：查找数组格式
                    array_pattern = r'\[[^\[\]]*\]'
                    matches = re.findall(array_pattern, content)
                    
                    for match in matches:
                        try:
                            return json.loads(match)
                        except:
                            continue
                    
                    raise ValueError(f"无法解析JSON文件: {file_name}")
                    
            except Exception as e:
                raise ValueError(f"加载文件失败 ({file_name}): {e}")
                
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    
    def _extract_losses(self, data, loss_data):
        """从数据中提取损失值"""
        # 尝试不同的数据结构
        if isinstance(data, dict):
            # 格式1: {'train_losses': [...], 'val_losses': [...]}
            if 'train_losses' in data and 'val_losses' in data:
                loss_data['train_losses'] = data['train_losses']
                loss_data['val_losses'] = data['val_losses']
                loss_data['epochs'] = list(range(1, len(data['train_losses']) + 1))
                return True
            
            # 格式2: {'epoch_1': {'train_loss': ..., 'val_loss': ...}, ...}
            epochs = [k for k in data.keys() if k.startswith('epoch_')]
            if epochs:
                epochs.sort(key=lambda x: int(x.split('_')[1]))
                for epoch in epochs:
                    epoch_data = data[epoch]
                    if 'train_loss' in epoch_data and 'val_loss' in epoch_data:
                        loss_data['train_losses'].append(epoch_data['train_loss'])
                        loss_data['val_losses'].append(epoch_data['val_loss'])
                        loss_data['epochs'].append(int(epoch.split('_')[1]))
                return len(loss_data['train_losses']) > 0
            
            # 格式3: {'history': {'loss': [...], 'val_loss': [...]}}
            if 'history' in data:
                history = data['history']
                if 'loss' in history and 'val_loss' in history:
                    loss_data['train_losses'] = history['loss']
                    loss_data['val_losses'] = history['val_loss']
                    loss_data['epochs'] = list(range(1, len(history['loss']) + 1))
                    return True
            
            # 格式4: 单值损失 {'final_train_loss': ..., 'final_val_loss': ...}
            final_loss_combinations = [
                ('final_train_loss', 'final_val_loss'),
                ('train_loss', 'val_loss'),
                ('training_loss', 'validation_loss'),
                ('best_train_loss', 'best_val_loss'),
                ('last_train_loss', 'last_val_loss')
            ]
            
            for train_key, val_key in final_loss_combinations:
                if train_key in data and val_key in data:
                    # 确保是数值类型
                    try:
                        train_loss = float(data[train_key])
                        val_loss = float(data[val_key])
                        loss_data['final_train_loss'] = train_loss
                        loss_data['final_val_loss'] = val_loss
                        print(f"   ✓ 找到最终损失: {train_key}={train_loss:.4f}, {val_key}={val_loss:.4f}")
                        return True
                    except (ValueError, TypeError):
                        continue
        
        # 格式4: 列表格式 [{'train_loss': ..., 'val_loss': ...}, ...]
        elif isinstance(data, list):
            for i, epoch_data in enumerate(data):
                if isinstance(epoch_data, dict):
                    if 'train_loss' in epoch_data and 'val_loss' in epoch_data:
                        loss_data['train_losses'].append(epoch_data['train_loss'])
                        loss_data['val_losses'].append(epoch_data['val_loss'])
                        loss_data['epochs'].append(i + 1)
            return len(loss_data['train_losses']) > 0
        
        return False
    
    def _extract_losses_from_checkpoint(self, checkpoint, loss_data):
        """从checkpoint中提取损失数据"""
        print(f"   🔍 检查checkpoint内容，可用键: {list(checkpoint.keys())}")
        
        # 常见的checkpoint键名（扩展更多可能的组合）
        loss_keys = [
            ('train_losses', 'val_losses'),
            ('train_loss_history', 'val_loss_history'),
            ('training_losses', 'validation_losses'),
            ('loss_history', 'val_loss_history'),
            ('train_loss_list', 'val_loss_list'),
            ('training_loss', 'validation_loss'),
            ('losses_train', 'losses_val'),
            ('train_losses_epoch', 'val_losses_epoch'),
            ('epoch_train_losses', 'epoch_val_losses'),
            ('history_train_loss', 'history_val_loss')
        ]
        
        # 尝试不同的键名组合
        for train_key, val_key in loss_keys:
            if train_key in checkpoint and val_key in checkpoint:
                train_losses = checkpoint[train_key]
                val_losses = checkpoint[val_key]
                
                # 确保是列表格式
                if isinstance(train_losses, (list, tuple)) and isinstance(val_losses, (list, tuple)):
                    if len(train_losses) > 0 and len(val_losses) > 0:
                        loss_data['train_losses'] = list(train_losses)
                        loss_data['val_losses'] = list(val_losses)
                        loss_data['epochs'] = list(range(1, len(train_losses) + 1))
                        print(f"   ✓ 找到损失历史: {train_key}, {val_key}")
                        return True
        
        # 尝试从嵌套的history字典中提取
        if 'history' in checkpoint:
            history = checkpoint['history']
            if isinstance(history, dict):
                # 尝试不同的键名
                train_keys = ['loss', 'train_loss', 'training_loss', 'train_losses']
                val_keys = ['val_loss', 'validation_loss', 'val_losses', 'valid_loss']
                
                for train_key in train_keys:
                    for val_key in val_keys:
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
        
        # 尝试从单个值获取最终损失
        single_loss_keys = [
            ('train_loss', 'val_loss'),
            ('training_loss', 'validation_loss'),
            ('final_train_loss', 'final_val_loss'),
            ('best_train_loss', 'best_val_loss'),
            ('last_train_loss', 'last_val_loss')
        ]
        
        for train_key, val_key in single_loss_keys:
            if train_key in checkpoint and val_key in checkpoint:
                loss_data['final_train_loss'] = float(checkpoint[train_key])
                loss_data['final_val_loss'] = float(checkpoint[val_key])
                print(f"   ✓ 找到最终损失: {train_key}={loss_data['final_train_loss']:.4f}, {val_key}={loss_data['final_val_loss']:.4f}")
                return True
        
        # 尝试从optimizer状态或其他可能的位置提取
        if 'optimizer' in checkpoint and isinstance(checkpoint['optimizer'], dict):
            optimizer_state = checkpoint['optimizer']
            if 'state' in optimizer_state and 'param_groups' in optimizer_state:
                # 有时损失信息可能存储在optimizer中
                pass
        
        print(f"   ⚠️ 未在checkpoint中找到损失数据")
        return False
    
    def _calculate_statistics(self, loss_data):
        """计算损失统计信息"""
        if loss_data['train_losses'] and loss_data['val_losses']:
            # 找到最佳验证损失
            val_losses = loss_data['val_losses']
            best_idx = np.argmin(val_losses)
            loss_data['best_epoch'] = loss_data['epochs'][best_idx]
            loss_data['best_val_loss'] = val_losses[best_idx]
            
            # 最终损失
            loss_data['final_train_loss'] = loss_data['train_losses'][-1]
            loss_data['final_val_loss'] = loss_data['val_losses'][-1]
            
            print(f"   📈 训练轮数: {len(loss_data['train_losses'])}")
            print(f"   🎯 最佳验证损失: {loss_data['best_val_loss']:.4f} (第{loss_data['best_epoch']}轮)")
            print(f"   📊 最终训练损失: {loss_data['final_train_loss']:.4f}")
            print(f"   📊 最终验证损失: {loss_data['final_val_loss']:.4f}")
        elif loss_data['final_train_loss'] is not None:
            print(f"   📊 训练损失: {loss_data['final_train_loss']:.4f}")
            print(f"   📊 验证损失: {loss_data['final_val_loss']:.4f}")
    
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
    
    def create_loss_comparison_plot(self, output_dir="loss_analysis"):
        """创建损失对比图"""
        if not self.models_data:
            print("❌ 没有模型数据可供绘图")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 过滤有完整损失历史的模型
        models_with_history = [m for m in self.models_data if m['train_losses']]
        
        if not models_with_history:
            print("❌ 没有模型包含完整的损失历史")
            return
        
        # 智能生成模型显示名称
        def get_display_name(model_name, max_length=40):
            """生成适合显示的模型名称：模型类型_训练模式"""
            # 提取模型类型和训练模式
            model_type = self._extract_model_type(model_name)
            training_mode = self._extract_training_mode(model_name)
            
            # 构建显示名称
            if training_mode:
                display_name = f"{model_type}_{training_mode}"
            else:
                display_name = model_type
            
            return display_name[:max_length]
        
        # 为每个模型生成显示名称
        display_names = {}
        for model in models_with_history:
            display_names[model['name']] = get_display_name(model['name'])
        
        # 检查重复名称并添加序号
        name_counts = {}
        for model_name, display_name in display_names.items():
            if display_name in name_counts:
                name_counts[display_name] += 1
                display_names[model_name] = f"{display_name}_{name_counts[display_name]}"
            else:
                name_counts[display_name] = 1
        
        # 创建更大的图形以容纳完整信息
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('模型训练损失对比分析', fontsize=18, fontweight='bold')
        
        # 1. 训练损失对比
        ax1 = axes[0, 0]
        for model in models_with_history:
            ax1.plot(model['epochs'], model['train_losses'], 
                    label=display_names[model['name']], linewidth=2)
        ax1.set_title('训练损失对比', fontsize=14)
        ax1.set_xlabel('训练轮数', fontsize=12)
        ax1.set_ylabel('训练损失', fontsize=12)
        # 调整图例位置和大小
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 验证损失对比
        ax2 = axes[0, 1]
        for model in models_with_history:
            ax2.plot(model['epochs'], model['val_losses'], 
                    label=display_names[model['name']], linewidth=2)
        ax2.set_title('验证损失对比', fontsize=14)
        ax2.set_xlabel('训练轮数', fontsize=12)
        ax2.set_ylabel('验证损失', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 最佳验证损失排名
        ax3 = axes[1, 0]
        models_sorted = sorted(models_with_history, key=lambda x: x['best_val_loss'])
        names = [display_names[m['name']] for m in models_sorted]
        best_losses = [m['best_val_loss'] for m in models_sorted]
        
        bars = ax3.bar(range(len(names)), best_losses, color='skyblue', alpha=0.7)
        ax3.set_title('最佳验证损失排名', fontsize=14)
        ax3.set_xlabel('模型', fontsize=12)
        ax3.set_ylabel('最佳验证损失', fontsize=12)
        ax3.set_xticks(range(len(names)))
        # 改进x轴标签显示
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
        
        # 智能添加数值标签，避免重叠
        self._add_smart_value_labels(ax3, bars, best_losses)
        
        # 4. 收敛性分析（最后10轮的损失变化）
        ax4 = axes[1, 1]
        for model in models_with_history:
            if len(model['val_losses']) >= 10:
                last_10 = model['val_losses'][-10:]
                epochs_last_10 = list(range(len(model['val_losses'])-9, len(model['val_losses'])+1))
                ax4.plot(epochs_last_10, last_10, 
                        label=display_names[model['name']], linewidth=2, marker='o')
        
        ax4.set_title('收敛性分析（最后10轮）', fontsize=14)
        ax4.set_xlabel('训练轮数', fontsize=12)
        ax4.set_ylabel('验证损失', fontsize=12)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 调整子图间距以防止重叠
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图像，使用更高的DPI和更好的边界设置
        plot_path = os.path.join(output_dir, 'loss_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📊 损失对比图已保存: {plot_path}")
        
        # 同时生成一个大尺寸版本用于详细查看
        plot_path_large = os.path.join(output_dir, 'loss_comparison_large.png')
        fig.set_size_inches(24, 18)
        plt.savefig(plot_path_large, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"📊 大尺寸损失对比图已保存: {plot_path_large}")
        
        plt.show()
    
    def _add_smart_value_labels(self, ax, bars, values, min_distance=0.05):
        """智能添加数值标签，避免重叠"""
        if not bars or not values:
            return
        
        # 计算图表的y轴范围
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # 计算最小距离（相对于y轴范围）
        min_distance_abs = min_distance * y_range
        
        # 收集所有标签的位置信息
        label_positions = []
        for bar, value in zip(bars, values):
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_height()
            label_positions.append({
                'x': x_pos,
                'y': y_pos,
                'value': value,
                'bar': bar,
                'adjusted_y': y_pos  # 调整后的y位置
            })
        
        # 检测和解决重叠
        num_labels = len(label_positions)
        if num_labels <= 1:
            # 单个标签，直接显示
            for label_info in label_positions:
                ax.text(label_info['x'], label_info['adjusted_y'],
                       f'{label_info["value"]:.3f}', 
                       ha='center', va='bottom', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            return
        
        # 多个标签时的智能布局
        # 方案1: 交替显示在上方和下方
        if num_labels <= 6:
            for i, label_info in enumerate(label_positions):
                if i % 2 == 0:
                    # 偶数索引：显示在柱子上方
                    y_pos = label_info['y'] + min_distance_abs * 0.5
                    va = 'bottom'
                else:
                    # 奇数索引：显示在柱子内部或下方
                    if label_info['y'] > y_range * 0.3:  # 柱子足够高
                        y_pos = label_info['y'] * 0.5  # 柱子中间
                        va = 'center'
                    else:
                        y_pos = label_info['y'] + min_distance_abs * 0.5
                        va = 'bottom'
                
                ax.text(label_info['x'], y_pos,
                       f'{label_info["value"]:.3f}', 
                       ha='center', va=va, fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # 方案2: 标签数量较多时，使用更紧凑的显示
        else:
            # 计算是否需要旋转标签
            font_size = max(7, 10 - num_labels // 3)  # 根据数量调整字体大小
            
            for i, label_info in enumerate(label_positions):
                # 使用阶梯式布局
                level = i % 3  # 三个层次
                y_offset = min_distance_abs * (0.5 + level * 0.8)
                y_pos = label_info['y'] + y_offset
                
                # 确保不超出图表范围
                if y_pos > y_max * 0.95:
                    y_pos = label_info['y'] * 0.7  # 放在柱子内部
                    va = 'center'
                    facecolor = 'yellow'
                else:
                    va = 'bottom'
                    facecolor = 'white'
                
                ax.text(label_info['x'], y_pos,
                       f'{label_info["value"]:.2f}',  # 减少小数位数
                       ha='center', va=va, fontsize=font_size,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=facecolor, alpha=0.8))
    
    def create_summary_table(self, output_dir="loss_analysis"):
        """创建汇总表格"""
        if not self.models_data:
            print("❌ 没有模型数据可供分析")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备表格数据
        table_data = []
        for model in self.models_data:
            row = {
                '模型名称': model['name'],
                '训练轮数': len(model['train_losses']) if model['train_losses'] else 'N/A',
                '最佳验证损失': f"{model['best_val_loss']:.4f}" if model['best_val_loss'] else 'N/A',
                '最佳轮数': model['best_epoch'] if model['best_epoch'] else 'N/A',
                '最终训练损失': f"{model['final_train_loss']:.4f}" if model['final_train_loss'] else 'N/A',
                '最终验证损失': f"{model['final_val_loss']:.4f}" if model['final_val_loss'] else 'N/A',
                '数据源': model['source_file'] or 'N/A'
            }
            table_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 按最佳验证损失排序
        df_sorted = df.copy()
        df_sorted['最佳验证损失_数值'] = pd.to_numeric(df_sorted['最佳验证损失'], errors='coerce')
        df_sorted = df_sorted.sort_values('最佳验证损失_数值').drop('最佳验证损失_数值', axis=1)
        
        # 保存CSV
        csv_path = os.path.join(output_dir, 'loss_summary.csv')
        df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"📋 汇总表格已保存: {csv_path}")
        
        # 打印表格
        print("\n📊 模型损失汇总表:")
        print("=" * 120)
        print(df_sorted.to_string(index=False))
        print("=" * 120)
        
        return df_sorted
    
    def generate_report(self, output_dir="loss_analysis"):
        """生成完整的分析报告"""
        print("\n🔍 生成损失分析报告...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建报告
        report_path = os.path.join(output_dir, 'loss_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 模型训练损失分析报告\n\n")
            f.write(f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**分析模型数量**: {len(self.models_data)}\n\n")
            
            # 总体统计
            if self.models_data:
                models_with_history = [m for m in self.models_data if m['train_losses']]
                if models_with_history:
                    best_model = min(models_with_history, key=lambda x: x['best_val_loss'])
                    f.write("## 总体统计\n\n")
                    f.write(f"- **最佳模型**: {best_model['name']}\n")
                    f.write(f"- **最佳验证损失**: {best_model['best_val_loss']:.4f}\n")
                    f.write(f"- **平均训练轮数**: {np.mean([len(m['train_losses']) for m in models_with_history]):.1f}\n\n")
            
            # 详细信息
            f.write("## 模型详细信息\n\n")
            for i, model in enumerate(self.models_data, 1):
                f.write(f"### {i}. {model['name']}\n\n")
                f.write(f"- **数据源**: {model['source_file']}\n")
                if model['train_losses']:
                    f.write(f"- **训练轮数**: {len(model['train_losses'])}\n")
                    f.write(f"- **最佳验证损失**: {model['best_val_loss']:.4f} (第{model['best_epoch']}轮)\n")
                    f.write(f"- **最终训练损失**: {model['final_train_loss']:.4f}\n")
                    f.write(f"- **最终验证损失**: {model['final_val_loss']:.4f}\n")
                else:
                    f.write(f"- **训练损失**: {model['final_train_loss']:.4f}\n")
                    f.write(f"- **验证损失**: {model['final_val_loss']:.4f}\n")
                f.write("\n")
        
        print(f"📄 分析报告已保存: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='批处理损失分析工具')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='模型目录的基础路径（默认为当前目录）')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='指定要分析的模型目录名称')
    parser.add_argument('--output_dir', type=str, default='loss_analysis',
                       help='输出目录（默认: loss_analysis）')
    parser.add_argument('--no_plot', action='store_true',
                       help='不生成图表')
    
    args = parser.parse_args()
    
    print("=== 批处理损失分析工具 ===")
    print(f"基础目录: {args.base_dir or os.getcwd()}")
    print(f"输出目录: {args.output_dir}")
    
    # 创建分析器
    analyzer = BatchLossAnalyzer(args.base_dir)
    
    # 确定要分析的模型目录
    if args.models:
        # 用户指定的模型
        base_dir = args.base_dir or os.getcwd()
        model_dirs = [os.path.join(base_dir, model) for model in args.models]
        model_dirs = [d for d in model_dirs if os.path.isdir(d)]
        print(f"\n🎯 用户指定 {len(model_dirs)} 个模型目录")
    else:
        # 自动检测
        model_dirs = None
    
    # 分析模型
    analyzer.analyze_models(model_dirs)
    
    if not analyzer.models_data:
        print("❌ 没有找到可分析的模型数据")
        return
    
    # 生成汇总表格
    analyzer.create_summary_table(args.output_dir)
    
    # 生成图表（如果有完整历史数据）
    if not args.no_plot:
        analyzer.create_loss_comparison_plot(args.output_dir)
    
    # 生成报告
    analyzer.generate_report(args.output_dir)
    
    print(f"\n🎉 分析完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 