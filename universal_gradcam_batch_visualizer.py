#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
跨平台Grad-CAM批处理可视化工具
=============================

支持平台：
- Windows 10/11
- Linux (Ubuntu, CentOS, etc.)
- macOS (理论支持)

功能：
- 批量处理多个模型
- 跨平台兼容
- 智能错误检测
- 详细进度报告
"""

import os
import sys
import subprocess
import json
import time
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

# 可选：过滤常见的兼容性警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-full backward hook.*deprecated.*")

class UniversalGradCAMVisualizer:
    def __init__(self, model_configs, visualization_params):
        self.model_configs = model_configs
        self.params = visualization_params
        self.start_time = datetime.now()
        self.results = []
        self.cwd = Path.cwd()
        self.platform_info = self._get_platform_info()
        
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
    
    def safe_print(self, message, use_emoji=True):
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
    
    def normalize_path(self, path):
        """跨平台路径标准化"""
        path_obj = Path(path)
        # 始终使用正斜杠，让Python自动适配平台
        return str(path_obj).replace('\\', '/')
    
    def validate_environment(self):
        """验证环境"""
        self.safe_print("验证环境...")
        
        # 检查Python环境
        try:
            import torch
            self.safe_print(f"Python环境: PyTorch {torch.__version__} | 平台: {self.platform_info['system']}")
        except ImportError:
            self.safe_print("错误: PyTorch未安装")
            return False
        
        # 检查可视化脚本
        gradcam_script = self.detect_gradcam_script_path()
        if not gradcam_script:
            self.safe_print("错误: 未找到Grad-CAM可视化脚本")
            return False
        
        return True
    
    def find_available_models(self):
        """查找可用的模型"""
        self.safe_print("扫描可用模型...")
        available_models = []
        
        for config in self.model_configs:
            model_path = config['model_path']
            if Path(model_path).exists():
                self.safe_print(f"  [找到] {config['name']}: {model_path}")
                available_models.append(config)
            else:
                self.safe_print(f"  [缺失] {config['name']}: {model_path}")
        
        return available_models
    
    def detect_gradcam_script_path(self):
        """检测Grad-CAM脚本路径"""
        possible_paths = [
            'src/enhanced_gradcam_visualization.py',
            'enhanced_gradcam_visualization.py',
            Path.cwd() / 'src' / 'enhanced_gradcam_visualization.py'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(path)
        
        return None
    
    def setup_python_path_for_cloud(self):
        """为云端环境设置Python路径"""
        cloud_indicators = [
            'autodl-tmp',
            'cloud_vgg_training_package',
            '/root/autodl-tmp',
            '/home/user'
        ]
        
        current_path = str(Path.cwd())
        is_cloud = any(indicator in current_path for indicator in cloud_indicators)
        
        if is_cloud:
            additional_paths = [
                str(Path.cwd() / 'src'),
                str(Path.cwd()),
                str(Path.cwd() / 'cloud_vgg_training_package' / 'src'),
            ]
            return additional_paths
        
        return []
    
    def generate_gradcam_command(self, config):
        """生成Grad-CAM命令"""
        # 检测脚本路径
        gradcam_script = self.detect_gradcam_script_path()
        
        # 云端环境特殊处理
        if IS_LINUX and any(indicator in str(Path.cwd()) for indicator in ['autodl-tmp', 'cloud_vgg_training_package']):
            # 优先使用修复版云端包装脚本
            fixed_wrapper_script = Path.cwd() / 'cloud_gradcam_wrapper_fixed.py'
            wrapper_script = Path.cwd() / 'cloud_gradcam_wrapper.py'
            
            if fixed_wrapper_script.exists():
                gradcam_script = str(fixed_wrapper_script)
                self.safe_print(f"   使用修复版包装器: {gradcam_script}")
            elif wrapper_script.exists():
                gradcam_script = str(wrapper_script)
                self.safe_print(f"   使用原版包装器: {gradcam_script}")
            else:
                self.safe_print(f"   未找到云端包装器，使用直接脚本: {gradcam_script}")
        
        cmd = [
            sys.executable,
            gradcam_script,
            '--model_path', config['model_path'],
            '--dataset', self.params['dataset_type'],
            '--bg_mode', config['bg_mode'],
            '--max_samples', str(config['max_samples'])
        ]
        
        if self.params['random_select']:
            cmd.append('--random_select')
        
        if self.params.get('feature_dataset'):
            cmd.extend(['--feature_dataset', self.params['feature_dataset']])
        
        return cmd
    
    def generate_unique_output_dir(self, config):
        """生成唯一的输出目录名"""
        # 基于修复后的可视化脚本生成预期目录名
        model_path_obj = Path(config['model_path'])
        model_filename = model_path_obj.stem
        parent_dir = model_path_obj.parent.name
        
        # 如果文件名是通用的，使用父目录信息
        if model_filename.lower() in ['best_model', 'model', 'final_model']:
            if parent_dir and parent_dir != '.':
                model_identifier = f"{parent_dir}_{model_filename}"
            else:
                model_identifier = model_filename
        else:
            model_identifier = model_filename
        
        # 生成输出目录名
        mode_suffix = config['bg_mode'] if config['bg_mode'] != 'all' else 'all'
        suffix = 'random' if self.params['random_select'] else 'ordered'
        
        output_dir = f'enhanced_gradcam_analysis_{model_identifier}_{mode_suffix}_{self.params["dataset_type"]}_{suffix}'
        return output_dir
    
    def find_actual_output_directory(self, config, expected_output_dir, stdout_text):
        """寻找实际生成的输出目录"""
        # 1. 首先检查预期目录
        if Path(expected_output_dir).exists():
            return expected_output_dir
        
        # 2. 从stdout输出中解析实际目录
        if stdout_text:
            lines = stdout_text.split('\n')
            for line in lines:
                # 寻找包含"查看结果"的行
                if '查看结果:' in line or 'output_dir:' in line.lower():
                    # 提取目录名
                    parts = line.split(':')
                    if len(parts) > 1:
                        dir_path = parts[1].strip().rstrip('/')
                        if Path(dir_path).exists():
                            return dir_path
        
        # 3. 搜索当前目录中符合模式的目录
        current_dir = Path.cwd()
        
        # 基于配置名称的搜索模式
        model_identifier = config['name']
        bg_mode = config['bg_mode']
        
        search_patterns = [
            f"enhanced_gradcam_analysis_{model_identifier}_*",
            f"enhanced_gradcam_analysis_*{model_identifier}*",
            f"enhanced_gradcam_analysis_*{bg_mode}*",
        ]
        
        for pattern in search_patterns:
            matching_dirs = list(current_dir.glob(pattern))
            if matching_dirs:
                # 返回最新的目录
                latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
                if latest_dir.is_dir():
                    return str(latest_dir)
        
        return None
    
    def run_single_visualization(self, config):
        """运行单个模型的可视化"""
        self.safe_print(f"开始可视化: {config['name']}")
        self.safe_print(f"   描述: {config['description']}")
        self.safe_print(f"   模型: {config['model_path']}")
        self.safe_print(f"   模式: {config['bg_mode']}")
        self.safe_print(f"   样本数: {config['max_samples']}")
        
        # 生成唯一的输出目录名
        expected_output_dir = self.generate_unique_output_dir(config)
        
        if self.params.get('skip_existing', False) and Path(expected_output_dir).exists():
            self.safe_print(f"   [跳过] 已存在的结果: {expected_output_dir}")
            return {
                'name': config['name'],
                'status': 'skipped',
                'output_dir': expected_output_dir,
                'message': '已存在结果目录'
            }
        
        # 生成命令
        cmd = self.generate_gradcam_command(config)
        
        if self.params.get('verbose', False):
            self.safe_print(f"   命令: {' '.join(cmd)}")
        
        # 记录执行前的目录状态
        before_dirs = set(Path.cwd().glob("enhanced_gradcam_analysis_*"))
        
        # 执行命令
        start_time = time.time()
        try:
            # 设置环境变量
            env = os.environ.copy()
            if IS_WINDOWS:
                env['PYTHONIOENCODING'] = 'utf-8'
            elif IS_LINUX:
                env['LANG'] = 'en_US.UTF-8'
                env['LC_ALL'] = 'en_US.UTF-8'
                
                # 为云端环境设置基本Python路径
                additional_paths = self.setup_python_path_for_cloud()
                if additional_paths:
                    current_python_path = env.get('PYTHONPATH', '')
                    new_python_path = ':'.join(additional_paths)
                    if current_python_path:
                        env['PYTHONPATH'] = f"{new_python_path}:{current_python_path}"
                    else:
                        env['PYTHONPATH'] = new_python_path
                    self.safe_print(f"设置PYTHONPATH: {env['PYTHONPATH']}")
            
            # 平台特定的subprocess配置
            subprocess_kwargs = {
                'capture_output': True,
                'text': True,
                'timeout': 300,  # 5分钟超时
                'env': env,
            }
            
            if IS_WINDOWS:
                subprocess_kwargs.update({
                    'encoding': 'utf-8',
                    'errors': 'ignore'
                })
            
            result = subprocess.run(cmd, **subprocess_kwargs)
            duration = time.time() - start_time
            
            if result.returncode == 0:
                self.safe_print(f"   [完成] ({duration:.1f}秒)")
                
                # 显示详细输出用于调试
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    self.safe_print(f"   标准输出 ({len(stdout_lines)} 行):")
                    # 显示最后10行输出
                    for line in stdout_lines[-10:]:
                        if line.strip():
                            self.safe_print(f"     {line}")
                
                if result.stderr:
                    stderr_lines = result.stderr.strip().split('\n')
                    self.safe_print(f"   标准错误 ({len(stderr_lines)} 行):")
                    # 显示前5行错误
                    for line in stderr_lines[:5]:
                        if line.strip():
                            self.safe_print(f"     {line}")
                
                # 严格验证输出目录是否真的存在
                actual_output_dir = self.find_actual_output_directory(config, expected_output_dir, result.stdout)
                
                # 检查是否有新目录生成
                after_dirs = set(Path.cwd().glob("enhanced_gradcam_analysis_*"))
                new_dirs = after_dirs - before_dirs
                
                success_indicators = []
                
                # 检查1: 预期目录是否存在
                if actual_output_dir and Path(actual_output_dir).exists():
                    success_indicators.append("预期目录存在")
                
                # 检查2: 是否有新目录生成
                if new_dirs:
                    success_indicators.append(f"生成了{len(new_dirs)}个新目录")
                    for new_dir in new_dirs:
                        self.safe_print(f"   新生成目录: {new_dir.name}")
                
                # 检查3: stdout是否包含成功关键词
                if result.stdout and any(keyword in result.stdout for keyword in ['完成', '成功', 'completed', 'success', '分析完成']):
                    success_indicators.append("输出包含成功关键词")
                
                # 检查4: 是否有真正的错误关键词（排除PyTorch警告）
                error_keywords = ['错误', 'failed', 'exception', 'traceback', 'modulenotfounderror', 'importerror']
                warning_keywords = ['userwarning', 'deprecat', 'warning']
                has_errors = False
                
                if result.stdout:
                    stdout_lower = result.stdout.lower()
                    for keyword in error_keywords:
                        if keyword in stdout_lower:
                            # 检查是否只是警告
                            if not any(warn in stdout_lower for warn in warning_keywords):
                                has_errors = True
                                break
                
                if result.stderr:
                    stderr_lower = result.stderr.lower()
                    for keyword in error_keywords:
                        if keyword in stderr_lower:
                            # 检查是否只是警告
                            if not any(warn in stderr_lower for warn in warning_keywords):
                                has_errors = True
                                break
                
                # 最终判断
                if success_indicators and not has_errors:
                    if actual_output_dir and Path(actual_output_dir).exists():
                        self.safe_print(f"   确认成功: {', '.join(success_indicators)}")
                        output_status = 'success'
                        output_message = '成功完成'
                    else:
                        self.safe_print(f"   部分成功: {', '.join(success_indicators)}")
                        output_status = 'partial_success'
                        output_message = '执行成功但输出目录位置不明'
                else:
                    self.safe_print(f"   伪成功检测: 返回码0但无实际输出")
                    if has_errors:
                        self.safe_print(f"   检测到错误信息")
                    output_status = 'false_success'
                    output_message = '执行返回成功但实际失败'
                
                return {
                    'name': config['name'],
                    'status': output_status,
                    'duration': duration,
                    'expected_output_dir': expected_output_dir,
                    'actual_output_dir': actual_output_dir,
                    'output_dir': actual_output_dir or expected_output_dir,
                    'samples': config['max_samples'],
                    'message': output_message,
                    'platform': self.platform_info['system']
                }
            else:
                self.safe_print(f"   [失败] ({duration:.1f}秒)")
                # 处理错误信息
                error_msg = result.stderr[:300] if result.stderr else "未知错误"
                if IS_WINDOWS:
                    error_msg = error_msg.encode('ascii', 'ignore').decode('ascii')
                self.safe_print(f"   错误: {error_msg}")
                
                return {
                    'name': config['name'],
                    'status': 'failed',
                    'duration': duration,
                    'error': error_msg,
                    'message': '执行失败',
                    'platform': self.platform_info['system']
                }
                
        except subprocess.TimeoutExpired:
            self.safe_print(f"   [超时] (>300秒)")
            return {
                'name': config['name'],
                'status': 'timeout',
                'message': '执行超时',
                'platform': self.platform_info['system']
            }
        except Exception as e:
            self.safe_print(f"   [异常] {str(e)}")
            return {
                'name': config['name'],
                'status': 'error',
                'error': str(e),
                'message': '执行异常',
                'platform': self.platform_info['system']
            }
    
    def run_batch_visualization(self):
        """运行批量可视化"""
        self.safe_print("开始批量Grad-CAM可视化")
        self.safe_print(f"配置的模型数量: {len(self.model_configs)}")
        self.safe_print(f"运行平台: {self.platform_info['system']} {self.platform_info['release']}")
        
        # 验证环境
        if not self.validate_environment():
            self.safe_print("环境验证失败，停止执行")
            return
        
        # 查找可用模型
        available_models = self.find_available_models()
        
        if not available_models:
            self.safe_print("没有找到可用的模型文件")
            return
        
        self.safe_print(f"将处理 {len(available_models)} 个模型:")
        for i, config in enumerate(available_models, 1):
            self.safe_print(f"  {i}. {config['name']} - {config['description']}")
        
        # 执行可视化
        self.safe_print("开始批量处理...")
        
        for i, config in enumerate(available_models, 1):
            self.safe_print("=" * 60)
            self.safe_print(f"处理进度: {i}/{len(available_models)}")
            
            result = self.run_single_visualization(config)
            self.results.append(result)
        
        # 生成汇总报告
        if self.params.get('generate_summary', True):
            self.generate_summary_report()
    
    def generate_summary_report(self):
        """生成汇总报告"""
        self.safe_print("=" * 60)
        self.safe_print("生成汇总报告")
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        successful = [r for r in self.results if r['status'] == 'success']
        partial_success = [r for r in self.results if r['status'] == 'partial_success']
        false_success = [r for r in self.results if r['status'] == 'false_success']
        completed_no_output = [r for r in self.results if r['status'] == 'completed_no_output']
        failed = [r for r in self.results if r['status'] == 'failed']
        skipped = [r for r in self.results if r['status'] == 'skipped']
        
        # 控制台报告
        self.safe_print("批量可视化完成!")
        self.safe_print(f"运行平台: {self.platform_info['system']}")
        self.safe_print(f"总耗时: {total_time:.1f}秒")
        self.safe_print(f"总模型: {len(self.results)}")
        self.safe_print(f"成功: {len(successful)}")
        if partial_success:
            self.safe_print(f"部分成功: {len(partial_success)}")
        if false_success:
            self.safe_print(f"伪成功: {len(false_success)}")
        if completed_no_output:
            self.safe_print(f"完成但无输出: {len(completed_no_output)}")
        self.safe_print(f"失败: {len(failed)}")
        self.safe_print(f"跳过: {len(skipped)}")
        
        if successful:
            self.safe_print("成功的模型:")
            for result in successful:
                duration = result.get('duration', 0)
                self.safe_print(f"  {result['name']} ({duration:.1f}秒)")
        
        if partial_success:
            self.safe_print("部分成功的模型:")
            for result in partial_success:
                duration = result.get('duration', 0)
                self.safe_print(f"  {result['name']} ({duration:.1f}秒)")
        
        if false_success:
            self.safe_print("伪成功的模型:")
            for result in false_success:
                duration = result.get('duration', 0)
                self.safe_print(f"  {result['name']} ({duration:.1f}秒)")
        
        if completed_no_output:
            self.safe_print("完成但无输出的模型:")
            for result in completed_no_output:
                duration = result.get('duration', 0)
                self.safe_print(f"  {result['name']} ({duration:.1f}秒)")
                if 'expected_output_dir' in result:
                    self.safe_print(f"     预期目录: {result['expected_output_dir']}")
        
        if failed:
            self.safe_print("失败的模型:")
            for result in failed:
                self.safe_print(f"  {result['name']}: {result.get('message', '未知错误')}")
        
        if skipped:
            self.safe_print("跳过的模型:")
            for result in skipped:
                self.safe_print(f"  {result['name']}: {result.get('message', '跳过')}")
        
        # 保存详细报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"gradcam_batch_report_universal_{timestamp}.json"
        
        report_data = {
            'timestamp': timestamp,
            'platform_info': self.platform_info,
            'total_time_seconds': total_time,
            'summary': {
                'total': len(self.results),
                'successful': len(successful),
                'partial_success': len(partial_success),
                'false_success': len(false_success),
                'completed_no_output': len(completed_no_output),
                'failed': len(failed),
                'skipped': len(skipped)
            },
            'results': self.results,
            'parameters': self.params
        }
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            self.safe_print(f"详细报告已保存: {report_filename}")
        except Exception as e:
            self.safe_print(f"保存报告失败: {e}")
        
        # 显示输出目录
        self.safe_print("可视化结果目录:")
        for result in successful:
            if 'actual_output_dir' in result and result['actual_output_dir']:
                self.safe_print(f"  {result['actual_output_dir']}/")
            elif 'output_dir' in result:
                self.safe_print(f"  {result['output_dir']}/")
        
        # 显示问题目录
        if false_success:
            self.safe_print("问题分析:")
            for result in false_success:
                self.safe_print(f"  {result['name']}: 执行完成但未生成预期输出")
                if 'expected_output_dir' in result:
                    self.safe_print(f"      预期: {result['expected_output_dir']}/")
                    
                    # 提供调试建议
                    self.safe_print("      建议检查:")
                    self.safe_print("      1. 脚本是否真正成功执行")
                    self.safe_print("      2. 输出目录命名逻辑是否正确")
                    self.safe_print("      3. 磁盘空间是否充足")
                    break  # 只显示一个示例，避免输出过长

def main():
    """主函数示例"""
    # 示例配置
    MODEL_CONFIGS = [
        {
            'name': 'test_model',
            'model_path': 'models/test_model.pth',
            'model_type': 'traditional_cnn',
            'bg_mode': 'bg0',
            'max_samples': 10,
            'description': '测试模型'
        }
    ]
    
    VISUALIZATION_PARAMS = {
        'dataset_type': 'feature',
        'random_select': True,
        'skip_existing': False,
        'verbose': True,
        'generate_summary': True,
        'feature_dataset': None
    }
    
    visualizer = UniversalGradCAMVisualizer(MODEL_CONFIGS, VISUALIZATION_PARAMS)
    visualizer.run_batch_visualization()

if __name__ == "__main__":
    main() 