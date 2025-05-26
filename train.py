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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)  # 确保目标值的形状正确
            
            optimizer.zero_grad()
            predictions, _ = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] 训练损失: {running_loss / 10:.3f}')
                running_loss = 0.0
        
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
        val_losses.append(val_loss)
        
        # 计算R²分数
        r2 = r2_score(val_targets, val_predictions)
        
        print(f'Epoch {epoch + 1} 验证损失: {val_loss:.3f}, R² 分数: {r2:.3f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型，验证损失: {val_loss:.3f}')
        
        # 绘制验证集上的预测值与真实值对比
        if epoch % 5 == 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(val_targets, val_predictions, alpha=0.5)
            plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
            plt.xlabel('真实浓度值')
            plt.ylabel('预测浓度值')
            plt.title(f'Epoch {epoch + 1} 预测值 vs 真实值 (R² = {r2:.3f})')
            plt.savefig(f'prediction_plot_epoch_{epoch + 1}.png')
            plt.close()

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            pred, _ = model(inputs)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return mse, r2, predictions, targets

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    dataset_path = r"D:\2025年实验照片"
    full_dataset = LocalImageDataset(root_dir=dataset_path, transform=transform)
    
    if len(full_dataset) == 0:
        print("错误：未找到任何有效的图像文件！")
        return
    
    # 划分数据集
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"数据集划分:")
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    print(f"测试集: {len(test_dataset)} 张图像")
    
    # 创建数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 创建模型实例
    model = CNNFeatureExtractor().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("\n开始训练模型...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_model.pth'))
    mse, r2, predictions, targets = evaluate_model(model, test_loader, device)
    
    print("\n测试集结果:")
    print(f"均方误差 (MSE): {mse:.3f}")
    print(f"R² 分数: {r2:.3f}")
    
    # 绘制最终的预测结果
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('真实浓度值')
    plt.ylabel('预测浓度值')
    plt.title(f'测试集预测结果 (R² = {r2:.3f})')
    plt.savefig('final_prediction_plot.png')
    plt.close()

if __name__ == '__main__':
    main() 