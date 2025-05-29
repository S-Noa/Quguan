import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from cnn_model import CNNFeatureExtractor
from custom_dataset import LocalImageDataset
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import argparse
import logging
import time
import sys
from datetime import datetime
from PIL import Image

# 导入字体设置工具
from font_utils import CHINESE_SUPPORTED, get_labels, suppress_font_warnings
# 导入训练监控模块
from training_monitor import TrainingMonitor

# 抑制字体警告
suppress_font_warnings()

def setup_logger(log_file=None, group_tag=None):
    """设置日志记录器"""
    # 创建日志文件名
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        group_suffix = f"_{group_tag}" if group_tag else ""
        log_file = f"cnn_training{group_suffix}_{timestamp}.log"
    
    # 创建logger
    logger = logging.getLogger('CNN_Training')
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

def log_system_info(logger):
    """记录系统信息"""
    logger.info("=== 系统信息 ===")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

def log_model_info(logger, model, device):
    """记录模型信息"""
    logger.info("=== 模型信息 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"模型大小: {total_params * 4 / 1024**2:.1f} MB")
    logger.info(f"模型设备: {device}")

def log_dataset_info(logger, dataset, dataset_name):
    """记录数据集信息"""
    logger.info(f"=== {dataset_name}数据集信息 ===")
    logger.info(f"数据集大小: {len(dataset)} 张图像")
    if hasattr(dataset, 'concentrations'):
        concentrations = dataset.concentrations
        logger.info(f"浓度范围: {min(concentrations):.1f} - {max(concentrations):.1f}")
        logger.info(f"平均浓度: {np.mean(concentrations):.2f}")
        logger.info(f"浓度标准差: {np.std(concentrations):.2f}")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, start_epoch=0, best_model_path='best_model.pth', group_tag=None, logger=None):
    """训练模型并记录详细日志"""
    if logger is None:
        logger, _ = setup_logger(group_tag=group_tag)
    
    logger.info("=== 开始训练 ===")
    logger.info(f"训练轮数: {num_epochs}")
    logger.info(f"起始轮数: {start_epoch}")
    logger.info(f"批次大小: {train_loader.batch_size}")
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    logger.info(f"优化器: {type(optimizer).__name__}")
    logger.info(f"学习率: {optimizer.param_groups[0]['lr']}")
    logger.info(f"损失函数: {type(criterion).__name__}")
    
    # 初始化训练监控器
    monitor_dir = f'monitoring_results_{group_tag}' if group_tag else 'monitoring_results_cnn'
    training_monitor = TrainingMonitor(model_type='cnn', save_dir=monitor_dir, device=device)
    logger.info(f"训练监控器已初始化，结果保存至: {monitor_dir}")
    
    # 获取一个样本图像用于监控
    sample_image = None
    sample_concentration = None
    try:
        for inputs, targets in val_loader:
            sample_image = inputs[0]  # 取第一张图像
            sample_concentration = targets[0].item()
            break
        logger.info(f"获取监控样本图像成功，浓度: {sample_concentration}")
    except Exception as e:
        logger.warning(f"获取监控样本图像失败: {e}")
    
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            predictions, _ = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            # 每10个batch记录一次
            if i % 10 == 9:
                avg_loss = running_loss / 10
                batch_time = time.time() - batch_start_time
                logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] 训练损失: {avg_loss:.6f}, 批次耗时: {batch_time:.2f}s')
                train_losses.append(avg_loss)
                running_loss = 0.0
                
                # GPU内存监控
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_cached = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f'GPU内存使用: {gpu_memory:.2f}GB / {gpu_cached:.2f}GB')
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float().view(-1, 1)
                predictions, _ = model(inputs)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        r2 = r2_score(val_targets, val_predictions)
        val_losses.append(val_loss)
        val_r2_scores.append(r2)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f'Epoch {epoch + 1} 完成 - 验证损失: {val_loss:.6f}, R² 分数: {r2:.6f}, 耗时: {epoch_time:.1f}s')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'✓ 保存最佳模型到 {best_model_path}，验证损失: {val_loss:.6f}')
        
        # 每5个epoch生成预测图和Grad-CAM监控
        if epoch % 5 == 0:
            try:
                # 获取标签
                labels = get_labels(CHINESE_SUPPORTED)
                
                # 生成预测对比图
                plt.figure(figsize=(10, 6))
                plt.scatter(val_targets, val_predictions, alpha=0.5)
                plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
                plt.xlabel(labels['true_concentration'])
                plt.ylabel(labels['predicted_concentration'])
                plt.title(f"{labels['epoch']} {epoch + 1} {labels['prediction_vs_true']} (R² = {r2:.3f})")
                prediction_plot_path = f'cnn_prediction_plot_epoch_{epoch + 1}.png'
                if group_tag:
                    prediction_plot_path = f'cnn_prediction_plot_{group_tag}_epoch_{epoch + 1}.png'
                plt.savefig(prediction_plot_path)
                plt.close()
                logger.info(f'✓ 预测对比图已保存: {prediction_plot_path}')
                
                # 生成Grad-CAM监控（如果有样本图像）
                if sample_image is not None:
                    logger.info(f"正在生成Epoch {epoch + 1}的Grad-CAM监控...")
                    analysis_result = training_monitor.visualize_gradcam_with_regions(
                        model, sample_image, epoch + 1, 
                        save_name=f'cnn_gradcam_{group_tag}_epoch_{epoch + 1}' if group_tag else f'cnn_gradcam_epoch_{epoch + 1}'
                    )
                    
                    if analysis_result:
                        # 记录监控结果到日志
                        regions_attention = analysis_result['regions_attention']
                        attention_ratios = analysis_result['attention_ratios']
                        
                        logger.info(f"=== Epoch {epoch + 1} 注意力分析 ===")
                        logger.info(f"中心区域注意力: {regions_attention['center']:.3f}")
                        logger.info(f"水印区域注意力(中): {regions_attention['watermark_medium']:.3f}")
                        logger.info(f"水印/中心注意力比例: {attention_ratios['medium_ratio']:.3f}")
                        
                        # 检查是否有警告
                        has_warning, warning_msg = training_monitor.check_watermark_attention(analysis_result)
                        if has_warning:
                            logger.warning(f"注意力警告: {warning_msg}")
                        else:
                            logger.info("✓ 注意力分布正常")
                
            except Exception as e:
                logger.error(f"生成监控可视化时出错: {e}")
    
    total_training_time = time.time() - training_start_time
    logger.info(f"\n=== 训练完成 ===")
    logger.info(f"总训练时间: {total_training_time:.1f}秒")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 生成最终监控报告
    try:
        report_name = f'cnn_training_monitoring_report_{group_tag}' if group_tag else 'cnn_training_monitoring_report'
        report_path = training_monitor.generate_monitoring_report(save_name=report_name)
        if report_path:
            logger.info(f"✓ 训练监控报告已生成: {report_path}")
    except Exception as e:
        logger.error(f"生成监控报告时出错: {e}")
    
    return best_val_loss

def evaluate_model(model, test_loader, device, logger=None):
    """评估模型并记录结果"""
    if logger is None:
        logger, _ = setup_logger()
    
    logger.info("=== 开始模型评估 ===")
    eval_start_time = time.time()
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for i, (inputs, target) in enumerate(test_loader):
            inputs = inputs.to(device)
            pred, _ = model(inputs)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.numpy())
            
            if i % 10 == 0:
                logger.info(f"评估进度: {i+1}/{len(test_loader)} 批次")
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算评估指标
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    mae = np.mean(np.abs(targets - predictions))
    rmse = np.sqrt(mse)
    
    eval_time = time.time() - eval_start_time
    
    logger.info(f"=== 评估结果 ===")
    logger.info(f"测试样本数: {len(targets)}")
    logger.info(f"均方误差 (MSE): {mse:.6f}")
    logger.info(f"均方根误差 (RMSE): {rmse:.6f}")
    logger.info(f"平均绝对误差 (MAE): {mae:.6f}")
    logger.info(f"R² 分数: {r2:.6f}")
    logger.info(f"评估耗时: {eval_time:.1f}s")
    
    # 预测值统计
    logger.info(f"真实值范围: {targets.min():.2f} - {targets.max():.2f}")
    logger.info(f"预测值范围: {predictions.min():.2f} - {predictions.max():.2f}")
    logger.info(f"预测误差范围: {(predictions-targets).min():.2f} - {(predictions-targets).max():.2f}")
    
    return mse, r2, predictions, targets

def main():
    parser = argparse.ArgumentParser(description='CNN分组训练开关')
    parser.add_argument('--group', type=str, default='all', choices=['all', 'bg0', 'bg1', 'allgroup'],
                        help='选择训练数据分组: all(全部), bg0(不补光), bg1(补光), allgroup(全部分组)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数 (默认: 15)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--data_path', type=str, default=r"D:\2025年实验照片_no_timestamp",
                        help='数据集路径（默认使用移除时间戳的干净数据集）')
    args = parser.parse_args()

    # 设置主日志记录器
    main_logger, log_file = setup_logger(group_tag=args.group)
    
    try:
        main_logger.info("=== CNN训练程序启动 ===")
        main_logger.info(f"命令行参数: {vars(args)}")
        main_logger.info(f"日志文件: {log_file}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_logger.info(f"使用设备: {device}")
        
        # 记录系统信息
        log_system_info(main_logger)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        main_logger.info(f"数据预处理: Resize(224,224) + ToTensor + Normalize")
        main_logger.info(f"数据路径: {args.data_path}")

        if args.group in ['all', 'allgroup']:
            # 全部数据训练
            main_logger.info("\n=== 【全部数据训练】 ===")
            
            try:
                full_dataset = LocalImageDataset(root_dir=args.data_path, transform=transform)
                if len(full_dataset) == 0:
                    main_logger.error("错误：未找到任何有效的图像文件！")
                    return
                
                log_dataset_info(main_logger, full_dataset, "完整")
                
                train_size = int(0.7 * len(full_dataset))
                val_size = int(0.15 * len(full_dataset))
                test_size = len(full_dataset) - train_size - val_size
                
                train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
                main_logger.info(f"数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")
                
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
                
                model = CNNFeatureExtractor().to(device)
                log_model_info(main_logger, model, device)
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                
                start_epoch = 0
                if os.path.exists('best_model.pth'):
                    main_logger.info('检测到已有best_model.pth，正在加载权重并断点续训...')
                    model.load_state_dict(torch.load('best_model.pth', map_location=device))
                    start_epoch = 2  # 假设已训练2轮
                
                train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                          num_epochs=args.epochs, start_epoch=start_epoch, group_tag='all', logger=main_logger)
                
                # 测试评估
                model.load_state_dict(torch.load('best_model.pth', map_location=device))
                mse, r2, predictions, targets = evaluate_model(model, test_loader, device, main_logger)
                
                # 生成最终预测图
                # 获取标签
                labels = get_labels(CHINESE_SUPPORTED)
                
                plt.figure(figsize=(10, 6))
                plt.scatter(targets, predictions, alpha=0.5)
                plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
                plt.xlabel(labels['true_concentration'])
                plt.ylabel(labels['predicted_concentration'])
                plt.title(f"{labels['all_data']}{labels['test_results']} (R² = {r2:.3f})")
                plt.savefig('cnn_final_prediction_plot_all.png')
                plt.close()
                main_logger.info('最终预测图已保存: cnn_final_prediction_plot_all.png')
                
            except Exception as e:
                main_logger.error(f"全部数据训练失败: {e}")
                import traceback
                main_logger.error(traceback.format_exc())

        if args.group in ['bg0', 'bg1', 'allgroup']:
            group_list = [args.group] if args.group in ['bg0', 'bg1'] else ['bg0', 'bg1']
            for bg_type in group_list:
                main_logger.info(f"\n=== 【仅{bg_type}图像训练】 ===")
                
                try:
                    dataset = LocalImageDataset(root_dir=args.data_path, transform=transform, bg_type=bg_type)
                    if len(dataset) < 10:
                        main_logger.warning(f"{bg_type}样本过少({len(dataset)}张)，跳过...")
                        continue
                    
                    log_dataset_info(main_logger, dataset, bg_type)
                    
                    train_size = int(0.7 * len(dataset))
                    val_size = int(0.15 * len(dataset))
                    test_size = len(dataset) - train_size - val_size
                    
                    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
                    main_logger.info(f"{bg_type}数据集划分: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")
                    
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
                    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
                    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
                    
                    model = CNNFeatureExtractor().to(device)
                    log_model_info(main_logger, model, device)
                    
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    
                    best_model_path = f'best_model_{bg_type}.pth'
                    start_epoch = 0
                    if os.path.exists(best_model_path):
                        main_logger.info(f'检测到已有{best_model_path}，正在加载权重并断点续训...')
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                        start_epoch = 1
                    
                    train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                              num_epochs=args.epochs, start_epoch=start_epoch, 
                              best_model_path=best_model_path, group_tag=bg_type, logger=main_logger)
                    
                    # 测试评估
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                    mse, r2, predictions, targets = evaluate_model(model, test_loader, device, main_logger)
                    
                    # 生成最终预测图
                    # 获取标签
                    labels = get_labels(CHINESE_SUPPORTED)
                    
                    plt.figure(figsize=(10, 6))
                    plt.scatter(targets, predictions, alpha=0.5)
                    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
                    plt.xlabel(labels['true_concentration'])
                    plt.ylabel(labels['predicted_concentration'])
                    plt.title(f'{bg_type}{labels["test_results"]} (R² = {r2:.3f})')
                    plt.savefig(f'cnn_final_prediction_plot_{bg_type}.png')
                    plt.close()
                    main_logger.info(f'{bg_type}最终预测图已保存: cnn_final_prediction_plot_{bg_type}.png')
                    
                except Exception as e:
                    main_logger.error(f"{bg_type}训练失败: {e}")
                    import traceback
                    main_logger.error(traceback.format_exc())
        
        main_logger.info("=== 程序执行完成 ===")
        
    except Exception as e:
        main_logger.error(f"程序执行失败: {e}")
        import traceback
        main_logger.error(traceback.format_exc())
    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            main_logger.info("已清理GPU缓存")

if __name__ == '__main__':
    main() 