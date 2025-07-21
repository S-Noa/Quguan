#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
跨平台Grad-CAM配置生成器
========================

支持平台：
- Windows 10/11
- Linux (Ubuntu, CentOS, etc.)
- macOS (理论支持)

功能：
- 自动扫描模型文件
- 生成跨平台配置
- 智能路径处理
- 交互式配置创建
"""

import os
import sys
import glob
import json
import platform
from datetime import datetime
from pathlib import Path

# 检测操作系统
IS_WINDOWS = platform.system().lower() == 'windows'
IS_LINUX = platform.system().lower() == 'linux'
IS_MACOS = platform.system().lower() == 'darwin'

# 设置编码环境
def setup_encoding():
    """设置合适的编码环境"""
    if IS_WINDOWS:
        try:
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            import locale
            try:
                locale.setlocale(locale.LC_ALL, 'C.UTF-8')
            except:
                pass
        except:
            pass
    elif IS_LINUX:
        os.environ.setdefault('LANG', 'en_US.UTF-8')
        os.environ.setdefault('LC_ALL', 'en_US.UTF-8')

# 初始化编码
setup_encoding()

def safe_print(message, use_emoji=True):
    """跨平台安全打印"""
    # 在Linux/macOS上保留emoji，Windows上移除
    if not use_emoji or IS_WINDOWS:
        # 移除emoji字符
        import re
        message = re.sub(r'[^\u0000-\u007F\u4e00-\u9fff]+', '', message)
    
    try:
        print(message)
    except UnicodeEncodeError:
        # 如果仍有编码错误，使用ASCII安全模式
        safe_message = message.encode('ascii', 'ignore').decode('ascii')
        print(safe_message)

def normalize_path(path):
    """跨平台路径标准化"""
    path_obj = Path(path)
    # 始终使用正斜杠，让Python自动适配平台
    return str(path_obj).replace('\\', '/')

class UniversalGradCAMConfigGenerator:
    def __init__(self):
        self.platform_info = self._get_platform_info()
        self.found_models = []
        self.selected_configs = []
        
    def _get_platform_info(self):
        """获取平台信息"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'is_windows': IS_WINDOWS,
            'is_linux': IS_LINUX,
            'is_macos': IS_MACOS
        }
    
    def scan_model_files(self):
        """扫描工作区中的模型文件"""
        safe_print(f"扫描模型文件... (平台: {self.platform_info['system']})")
        
        # 定义搜索模式
        search_patterns = [
            '**/best_model.pth',
            '**/best_model_*.pth',
            '**/*_model.pth',
            '**/model_*.pth',
            'models/**/*.pth',
            '*.pth'
        ]
        
        # 排除目录
        exclude_patterns = [
            'yolo*',
            'archive*',
            'deprecated*',
            '__pycache__',
            '.git',
            'logs',
            'temp*'
        ]
        
        found_files = set()
        
        for pattern in search_patterns:
            files = Path('.').glob(pattern)
            for file_path in files:
                if file_path.is_file():
                    # 检查是否在排除目录中
                    should_exclude = False
                    for exclude in exclude_patterns:
                        if any(exclude.lower() in part.lower() for part in file_path.parts):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        # 跨平台路径标准化
                        normalized_path = normalize_path(file_path)
                        found_files.add(normalized_path)
        
        # 转换为列表并排序
        self.found_models = sorted(list(found_files))
        safe_print(f"找到 {len(self.found_models)} 个模型文件")
        
        return self.found_models
    
    def analyze_model_info(self, model_path):
        """分析模型信息并推断配置"""
        path_lower = model_path.lower()
        file_name = Path(model_path).name.lower()
        parent_dir = Path(model_path).parent.name.lower() if Path(model_path).parent.name else ""
        
        # 分析模型类型
        model_type = 'unknown'
        if 'enhanced' in path_lower:
            if 'vgg' in path_lower:
                model_type = 'enhanced_laser_spot_cnn_vgg'
            elif 'resnet' in path_lower:
                model_type = 'enhanced_laser_spot_cnn_resnet'
            else:
                model_type = 'enhanced_laser_spot_cnn_vgg'
        elif 'vgg' in path_lower:
            model_type = 'vgg'
        elif 'resnet' in path_lower:
            model_type = 'resnet50'
        elif 'baseline' in path_lower or 'cnn' in path_lower:
            model_type = 'traditional_cnn'
        
        # 分析背景模式
        bg_mode = 'all'
        if 'bg0' in path_lower and 'bg1' not in path_lower:
            # 检查功率级别
            if '20mw' in path_lower:
                bg_mode = 'bg0_20mw'
            elif '100mw' in path_lower:
                bg_mode = 'bg0_100mw'
            elif '400mw' in path_lower:
                bg_mode = 'bg0_400mw'
            else:
                bg_mode = 'bg0'
        elif 'bg1' in path_lower and 'bg0' not in path_lower:
            if '20mw' in path_lower:
                bg_mode = 'bg1_20mw'
            elif '100mw' in path_lower:
                bg_mode = 'bg1_100mw'
            elif '400mw' in path_lower:
                bg_mode = 'bg1_400mw'
            else:
                bg_mode = 'bg1'
        
        # 生成唯一的配置名称
        name_parts = []
        if model_type != 'unknown':
            name_parts.append(model_type)
        if bg_mode != 'all':
            name_parts.append(bg_mode)
            
        # 添加路径特征以确保唯一性
        if parent_dir and parent_dir != '.':
            # 从父目录中提取时间戳或其他标识信息
            import re
            # 提取时间戳作为唯一标识
            timestamp_match = re.search(r'(\d{8}_\d{6})', parent_dir)
            if timestamp_match:
                name_parts.append(timestamp_match.group(1))
            else:
                # 如果没有时间戳，使用父目录的关键部分
                if 'results' in parent_dir:
                    # 移除results后缀，保留其他特征
                    clean_parent = parent_dir.replace('_results', '').replace('results_', '')
                    # 提取版本或其他标识
                    version_match = re.search(r'(v\d+|\d{6,8})', clean_parent)
                    if version_match:
                        name_parts.append(version_match.group(1))
                    elif len(clean_parent) > 0:
                        # 使用清理后的父目录名的最后一部分
                        name_parts.append(clean_parent.split('_')[-1])
        
        # 如果文件名不是标准的best_model，也加入名称
        file_stem = Path(model_path).stem
        if file_stem not in ['best_model', 'latest_model', 'model']:
            name_parts.append(file_stem)
        
        config_name = '_'.join(name_parts) if name_parts else Path(model_path).stem
        
        # 确保名称长度合理
        if len(config_name) > 80:
            # 保留前缀和重要的区分信息
            if len(name_parts) > 3:
                config_name = '_'.join([name_parts[0], name_parts[1], name_parts[-1]])
            else:
                config_name = config_name[:80]
        
        # 生成描述
        type_desc = {
            'traditional_cnn': '传统CNN',
            'advanced_cnn': '高级CNN', 
            'vgg': 'VGG16',
            'resnet50': 'ResNet50',
            'enhanced_laser_spot_cnn_vgg': '增强激光光斑CNN-VGG',
            'enhanced_laser_spot_cnn_resnet': '增强激光光斑CNN-ResNet'
        }.get(model_type, '未知模型')
        
        bg_desc = {
            'bg0': 'BG0背景',
            'bg1': 'BG1背景',
            'bg0_20mw': 'BG0背景-20mW',
            'bg0_100mw': 'BG0背景-100mW',
            'bg0_400mw': 'BG0背景-400mW',
            'bg1_20mw': 'BG1背景-20mW',
            'bg1_100mw': 'BG1背景-100mW',
            'bg1_400mw': 'BG1背景-400mW'
        }.get(bg_mode, '全部模式')
        
        description = f"{type_desc} - {bg_desc}"
        
        return {
            'name': config_name,
            'model_path': model_path,
            'model_type': model_type,
            'bg_mode': bg_mode,
            'description': description,
            'max_samples': 15
        }
    
    def display_found_models(self):
        """显示找到的模型"""
        safe_print("发现的模型文件:")
        safe_print("=" * 80)
        
        for i, model_path in enumerate(self.found_models, 1):
            info = self.analyze_model_info(model_path)
            safe_print(f"{i:2d}. {info['name']}")
            safe_print(f"    路径: {model_path}")
            safe_print(f"    类型: {info['model_type']}")
            safe_print(f"    模式: {info['bg_mode']}")
            safe_print(f"    描述: {info['description']}")
            safe_print("")  # 空行
    
    def generate_full_config(self):
        """生成所有模型的配置"""
        safe_print("生成全部模型配置...")
        
        for model_path in self.found_models:
            config = self.analyze_model_info(model_path)
            self.selected_configs.append(config)
        
        safe_print(f"已生成 {len(self.selected_configs)} 个配置")
    
    def interactive_model_selection(self):
        """交互式模型选择"""
        safe_print("交互式模型选择")
        safe_print("输入模型编号 (用空格分隔多个编号，例如: 1 3 5-8)")
        safe_print("输入 'all' 选择全部，输入 'quit' 退出")
        
        while True:
            try:
                user_input = input("请选择模型编号: ").strip()
                
                if user_input.lower() == 'quit':
                    return
                elif user_input.lower() == 'all':
                    self.generate_full_config()
                    return
                
                # 解析输入
                selected_indices = []
                parts = user_input.split()
                
                for part in parts:
                    if '-' in part:
                        # 范围选择，例如 5-8
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start, end + 1))
                    else:
                        # 单个选择
                        selected_indices.append(int(part))
                
                # 验证并添加配置
                valid_indices = [i for i in selected_indices if 1 <= i <= len(self.found_models)]
                
                if valid_indices:
                    for i in valid_indices:
                        model_path = self.found_models[i-1]
                        config = self.analyze_model_info(model_path)
                        if config not in self.selected_configs:
                            self.selected_configs.append(config)
                    
                    safe_print(f"已选择 {len(valid_indices)} 个模型")
                    return
                else:
                    safe_print("没有有效的选择，请重新输入")
                    
            except (ValueError, IndexError) as e:
                safe_print(f"输入格式错误: {e}")
                safe_print("请使用正确的格式，例如: 1 3 5-8")
    
    def generate_quick_config(self):
        """生成快速常用配置"""
        safe_print("生成快速常用配置...")
        
        # 寻找常用模型模式
        priority_patterns = [
            'best_model_bg0.pth',
            'enhanced_cnn_vgg16',
            'enhanced_cnn_resnet50',
            'baseline_cnn',
            'bg0_100mw',
            'bg1_100mw'
        ]
        
        quick_models = []
        for pattern in priority_patterns:
            for model_path in self.found_models:
                if pattern.lower() in model_path.lower() and model_path not in quick_models:
                    quick_models.append(model_path)
                    if len(quick_models) >= 6:  # 限制数量
                        break
            if len(quick_models) >= 6:
                break
        
        # 如果快速模式找到的不够，补充其他模型
        if len(quick_models) < 6:
            for model_path in self.found_models:
                if model_path not in quick_models:
                    quick_models.append(model_path)
                    if len(quick_models) >= 6:
                        break
        
        for model_path in quick_models:
            config = self.analyze_model_info(model_path)
            self.selected_configs.append(config)
        
        safe_print(f"快速配置已生成 {len(self.selected_configs)} 个模型")
    
    def save_config_file(self, filename=None):
        """保存配置文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'universal_gradcam_config_{timestamp}.py'
        
        # 生成跨平台配置内容
        config_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
跨平台Grad-CAM批处理配置
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
生成平台: {self.platform_info['system']} {self.platform_info['release']}
模型数量: {len(self.selected_configs)}
"""

# ================================
# 模型配置区域 - 跨平台路径
# ================================

MODEL_CONFIGS = [
'''
        
        for config in self.selected_configs:
            # 确保路径使用正斜杠，Python会自动适配平台
            normalized_path = normalize_path(config['model_path'])
            
            config_content += f'''    # {config['description']}
    {{
        'name': '{config['name']}',
        'model_path': '{normalized_path}',
        'bg_mode': '{config['bg_mode']}',
        'max_samples': {config['max_samples']},
        'description': '{config['description']}'
    }},
    
'''
        
        config_content += ''']

# ================================
# 可视化参数配置
# ================================

VISUALIZATION_PARAMS = {
    'dataset_type': 'feature',
    'random_select': True,
    'feature_dataset': None,
    'generate_summary': True,
    'save_configs': True,
    'skip_existing': True,
    'verbose': True,
    'backup_results': False,
}

if __name__ == "__main__":
    # 导入主可视化器
    import sys
    sys.path.append('.')
    
    try:
        from universal_gradcam_batch_visualizer import UniversalGradCAMVisualizer
        
        print("使用跨平台配置启动Grad-CAM批处理可视化...")
        visualizer = UniversalGradCAMVisualizer(MODEL_CONFIGS, VISUALIZATION_PARAMS)
        visualizer.run_batch_visualization()
        
    except ImportError:
        print("错误: 找不到 universal_gradcam_batch_visualizer.py")
        print("请确保该文件在当前目录中")
'''
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(config_content)
            safe_print(f"配置文件已保存: {filename}")
            
            # 创建快速运行脚本
            self.create_quick_run_script(filename)
            
            return filename
        except Exception as e:
            safe_print(f"保存配置文件失败: {e}")
            return None
    
    def create_quick_run_script(self, config_filename):
        """创建快速运行脚本"""
        script_name = config_filename.replace('.py', '_run.py')
        
        script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速运行脚本 - 跨平台
配置文件: {config_filename}
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import sys
import platform

def main():
    print(f"跨平台Grad-CAM快速运行脚本")
    print(f"运行平台: {{platform.system()}} {{platform.release()}}")
    print(f"配置文件: {config_filename}")
    print("=" * 60)
    
    try:
        # 导入配置
        import {config_filename.replace('.py', '')} as config
        from universal_gradcam_batch_visualizer import UniversalGradCAMVisualizer
        
        # 运行可视化
        visualizer = UniversalGradCAMVisualizer(
            config.MODEL_CONFIGS, 
            config.VISUALIZATION_PARAMS
        )
        visualizer.run_batch_visualization()
        
    except ImportError as e:
        print(f"导入错误: {{e}}")
        print("请确保以下文件存在:")
        print(f"  - {config_filename}")
        print("  - universal_gradcam_batch_visualizer.py")
    except Exception as e:
        print(f"运行错误: {{e}}")

if __name__ == "__main__":
    main()
'''
        
        try:
            with open(script_name, 'w', encoding='utf-8') as f:
                f.write(script_content)
            safe_print(f"快速运行脚本已创建: {script_name}")
        except Exception as e:
            safe_print(f"创建快速运行脚本失败: {e}")
    
    def save_summary_report(self):
        """保存配置摘要报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'universal_gradcam_config_summary_{timestamp}.json'
        
        report_data = {
            'timestamp': timestamp,
            'platform_info': self.platform_info,
            'scan_results': {
                'total_models_found': len(self.found_models),
                'selected_models': len(self.selected_configs),
                'found_model_paths': self.found_models
            },
            'selected_configs': self.selected_configs
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            safe_print(f"配置摘要已保存: {report_file}")
        except Exception as e:
            safe_print(f"保存摘要失败: {e}")

def main():
    """主函数"""
    safe_print("跨平台Grad-CAM配置生成器")
    safe_print(f"运行平台: {platform.system()} {platform.release()}")
    safe_print("=" * 60)
    
    generator = UniversalGradCAMConfigGenerator()
    
    # 扫描模型
    models = generator.scan_model_files()
    
    if not models:
        safe_print("没有找到模型文件！")
        return
    
    # 显示找到的模型
    generator.display_found_models()
    
    # 选择生成模式
    safe_print("选择生成模式:")
    safe_print("1. 自动生成全部模型配置")
    safe_print("2. 交互式选择模型")
    safe_print("3. 快速常用配置")
    safe_print("4. 退出")
    
    try:
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == '1':
            generator.generate_full_config()
        elif choice == '2':
            generator.interactive_model_selection()
        elif choice == '3':
            generator.generate_quick_config()
        elif choice == '4':
            safe_print("退出程序")
            return
        else:
            safe_print("无效选择，使用快速配置模式")
            generator.generate_quick_config()
        
        if generator.selected_configs:
            # 保存配置
            config_file = generator.save_config_file()
            generator.save_summary_report()
            
            safe_print("=" * 60)
            safe_print("配置生成完成!")
            safe_print(f"选择的模型数量: {len(generator.selected_configs)}")
            safe_print(f"配置文件: {config_file}")
            safe_print("使用方法:")
            safe_print(f"  python {config_file}")
            safe_print("  或运行对应的快速脚本")
        else:
            safe_print("没有选择任何模型")
            
    except KeyboardInterrupt:
        safe_print("\\n用户取消操作")
    except Exception as e:
        safe_print(f"操作失败: {e}")

if __name__ == "__main__":
    main() 