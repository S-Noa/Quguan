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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, start_epoch=0):
    best_val_loss = float('inf')
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            predictions, _ = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] 训练损失: {running_loss / 10:.3f}')
                running_loss = 0.0
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
        print(f'Epoch {epoch + 1} 验证损失: {val_loss:.3f}, R² 分数: {r2:.3f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'保存最佳模型，验证损失: {val_loss:.3f}')
        if epoch % 5 == 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(val_targets, val_predictions, alpha=0.5)
            plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
            plt.xlabel('真实浓度值')
            plt.ylabel('预测浓度值')
            plt.title(f'Epoch {epoch + 1} 预测值 vs 真实值 (R² = {r2:.3f})')
            plt.savefig(f'cnn_prediction_plot_epoch_{epoch + 1}.png')
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    dataset_path = r"D:\2025年实验照片"
    full_dataset = LocalImageDataset(root_dir=dataset_path, transform=transform)
    if len(full_dataset) == 0:
        print("错误：未找到任何有效的图像文件！")
        return
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
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = CNNFeatureExtractor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_epoch = 2
    # 断点续训：如果有best_model.pth则加载
    if os.path.exists('best_model.pth'):
        print('检测到已有best_model.pth，正在加载权重并断点续训...')
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        # 可选：加载上次训练到的epoch数（如有保存）
    print("\n开始训练模型...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15, start_epoch=start_epoch)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    mse, r2, predictions, targets = evaluate_model(model, test_loader, device)
    print("\n测试集结果:")
    print(f"均方误差 (MSE): {mse:.3f}")
    print(f"R² 分数: {r2:.3f}")
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('真实浓度值')
    plt.ylabel('预测浓度值')
    plt.title(f'测试集预测结果 (R² = {r2:.3f})')
    plt.savefig('cnn_final_prediction_plot.png')
    plt.close()

if __name__ == '__main__':
    main() 