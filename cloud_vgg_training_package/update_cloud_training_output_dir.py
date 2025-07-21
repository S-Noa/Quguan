#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端训练输出目录更新脚本
为train_with_feature_dataset.py添加基于数据集的输出目录命名功能
"""

import os
import re

def update_cloud_training_script():
    """更新云端训练脚本的输出目录设置"""
    script_path = "train_with_feature_dataset.py"
    
    if not os.path.exists(script_path):
        print(f"错误：找不到云端训练脚本: {script_path}")
        return False
    
    # 读取原始文件
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 在导入部分添加数据集名称工具
    import_pattern = r'(# 添加src目录到路径\nsys\.path\.append\(\'src\'\))'
    import_replacement = r'''\1

# 添加数据集名称工具
try:
    from dataset_name_utils import generate_training_output_dir, parse_training_mode_from_args, get_dataset_info_string
    DATASET_UTILS_AVAILABLE = True
    print("数据集名称工具加载成功")
except ImportError:
    print("警告：无法导入数据集名称工具，将使用传统命名方式")
    DATASET_UTILS_AVAILABLE = False'''
    
    if '# 添加数据集名称工具' not in content:
        content = re.sub(import_pattern, import_replacement, content)
        print("✅ 添加数据集名称工具导入")
    else:
        print("ℹ️ 数据集名称工具导入已存在")
    
    # 2. 修改输出目录生成逻辑
    output_dir_pattern = r'''(        # 设置输出目录，支持6档细分命名 \+ 冻结状态区分\s+timestamp = datetime\.now\(\)\.strftime\("%Y%m%d_%H%M%S"\)\s+mode_suffix = args\.bg_mode\.replace\('_', '-'\)  # bg0_20mw -> bg0-20mw\s+# 添加冻结状态标识\s+freeze_suffix = self\._get_freeze_suffix\(args\)\s+self\.output_dir = f"feature_training_{args\.model_type}_{mode_suffix}{freeze_suffix}_results_{timestamp}"\s+os\.makedirs\(self\.output_dir, exist_ok=True\))'''
    
    output_dir_replacement = r'''        # 设置输出目录，优先使用数据集名称工具
        if DATASET_UTILS_AVAILABLE:
            try:
                # 解析训练模式
                bg_filter, power_filter, training_mode = parse_training_mode_from_args(args)
                
                # 添加冻结状态标识
                freeze_suffix = self._get_freeze_suffix(args)
                model_name_with_suffix = f"{args.model_type}{freeze_suffix}"
                
                # 生成基于数据集的输出目录
                dataset_path = getattr(args, 'feature_dataset_path', None)
                if dataset_path:
                    self.output_dir = generate_training_output_dir(
                        model_name=model_name_with_suffix,
                        dataset_path=dataset_path,
                        training_mode=training_mode,
                        bg_filter=bg_filter,
                        power_filter=power_filter
                    )
                    os.makedirs(self.output_dir, exist_ok=True)
                    print(f"使用基于数据集的输出目录: {self.output_dir}")
                else:
                    # 如果没有数据集路径，使用传统方式
                    raise ValueError("数据集路径未指定")
                    
            except Exception as e:
                # 回退到传统方式
                print(f"生成基于数据集的输出目录失败: {e}，使用传统方式")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mode_suffix = args.bg_mode.replace('_', '-')
                freeze_suffix = self._get_freeze_suffix(args)
                self.output_dir = f"feature_training_{args.model_type}_{mode_suffix}{freeze_suffix}_results_{timestamp}"
                os.makedirs(self.output_dir, exist_ok=True)'''
    
    # 使用更简单的模式匹配
    if 'DATASET_UTILS_AVAILABLE' not in content:
        # 查找输出目录设置的开始位置
        pattern = r'(        # 设置输出目录，支持6档细分命名 \+ 冻结状态区分.*?os\.makedirs\(self\.output_dir, exist_ok=True\))'
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, output_dir_replacement, content, flags=re.DOTALL)
            print("✅ 更新输出目录生成逻辑")
        else:
            print("⚠️ 未找到输出目录设置模式，请手动检查")
    else:
        print("ℹ️ 输出目录生成逻辑已更新")
    
    # 3. 在日志部分添加数据集信息
    log_pattern = r'(        self\.logger\.info\(f"   输出目录: {self\.output_dir}"\))'
    log_replacement = r'''\1
        
        # 添加数据集信息到日志
        if DATASET_UTILS_AVAILABLE:
            try:
                dataset_path = getattr(args, 'feature_dataset_path', None)
                if dataset_path:
                    bg_filter, power_filter, _ = parse_training_mode_from_args(args)
                    dataset_info = get_dataset_info_string(dataset_path, bg_filter, power_filter)
                    self.logger.info(f"   {dataset_info}")
            except:
                pass'''
    
    if '# 添加数据集信息到日志' not in content:
        content = re.sub(log_pattern, log_replacement, content)
        print("✅ 添加数据集信息到日志")
    else:
        print("ℹ️ 数据集信息日志已存在")
    
    # 写回文件
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 云端训练脚本更新完成: {script_path}")
    return True

def main():
    print("更新云端训练脚本的输出目录设置...")
    
    # 确保数据集名称工具存在
    if not os.path.exists("src/dataset_name_utils.py"):
        print("复制数据集名称工具...")
        import shutil
        if os.path.exists("../dataset_name_utils.py"):
            shutil.copy2("../dataset_name_utils.py", "src/dataset_name_utils.py")
            print("✅ 数据集名称工具已复制")
        else:
            print("❌ 找不到数据集名称工具")
            return
    
    # 更新云端训练脚本
    if update_cloud_training_script():
        print("\n🎉 云端训练脚本更新成功！")
        print("\n现在可以使用以下命令进行训练:")
        print("python train_with_feature_dataset.py --model_type vgg --bg_mode all")
        print("python train_with_feature_dataset.py --model_type resnet50 --bg_mode bg0_400mw")
        print("python train_with_feature_dataset.py --model_type cnn --bg_mode bg1")
        print("\n输出目录将自动根据数据集和训练模式命名！")
    else:
        print("❌ 云端训练脚本更新失败")

if __name__ == "__main__":
    main() 