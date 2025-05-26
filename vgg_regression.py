import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from custom_dataset import LocalImageDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os

class VGGRegression(nn.Module):
    def __init__(self, freeze_features=True):
        super(VGGRegression, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.reg_head(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, start_epoch=0, use_amp=True):
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
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
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions = model(inputs)
                        loss = criterion(predictions, targets)
                else:
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        val_loss /= len(val_loader)
        r2 = r2_score(val_targets, val_predictions)
        print(f'Epoch {epoch + 1} 验证损失: {val_loss:.3f}, R² 分数: {r2:.3f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_vgg_model.pth')
            print(f'保存最佳模型，验证损失: {val_loss:.3f}')
        if epoch % 5 == 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(val_targets, val_predictions, alpha=0.5)
            plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
            plt.xlabel('真实浓度值')
            plt.ylabel('预测浓度值')
            plt.title(f'Epoch {epoch + 1} 预测值 vs 真实值 (R² = {r2:.3f})')
            plt.savefig(f'vgg_prediction_plot_epoch_{epoch + 1}.png')
            plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 数据增强
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_path = r"D:\2025年实验照片"
    full_dataset = LocalImageDataset(root_dir=dataset_path, transform=transform)
    if len(full_dataset) == 0:
        print("错误：未找到任何有效的图像文件！")
        return
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    print(f"数据集划分:")
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    print(f"测试集: {len(test_dataset)} 张图像")
    batch_size = 96
    num_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = VGGRegression(freeze_features=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.reg_head.parameters(), lr=0.001)
    start_epoch = 0
    # 断点续训：如果有best_vgg_model.pth则加载
    if os.path.exists('best_vgg_model.pth'):
        print('检测到已有best_vgg_model.pth，正在加载权重并断点续训...')
        model.load_state_dict(torch.load('best_vgg_model.pth', map_location=device))
        # 可选：加载上次训练到的epoch数（如有保存）
    print("\n开始训练模型...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, start_epoch=start_epoch, use_amp=True)
    model.load_state_dict(torch.load('best_vgg_model.pth', map_location=device))
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            with torch.cuda.amp.autocast():
                predictions = model(inputs)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(targets.numpy())
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    mse = mean_squared_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)
    print("\n测试集结果:")
    print(f"均方误差 (MSE): {mse:.3f}")
    print(f"R² 分数: {r2:.3f}")
    plt.figure(figsize=(10, 6))
    plt.scatter(test_targets, test_predictions, alpha=0.5)
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
    plt.xlabel('真实浓度值')
    plt.ylabel('预测浓度值')
    plt.title(f'测试集预测结果 (R² = {r2:.3f})')
    plt.savefig('vgg_final_prediction_plot.png')
    plt.close()

if __name__ == '__main__':
    main() 