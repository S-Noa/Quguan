import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from cnn_model import CNNFeatureExtractor
from custom_dataset import LocalImageDataset
import os
import numpy as np

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            # 使用随机标签进行训练，因为我们只关心特征提取
            random_labels = torch.randint(0, 10, (inputs.size(0),)).to(device)
            loss = criterion(outputs, random_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:  # 更频繁地打印损失
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.3f}')
                running_loss = 0.0

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    file_paths = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            _, batch_features = model(inputs)
            features.append(batch_features.cpu())
            
            # 获取当前批次的文件路径
            start_idx = i * dataloader.batch_size
            end_idx = start_idx + inputs.size(0)
            file_paths.extend(dataloader.dataset.image_files[start_idx:end_idx])
            
            if i % 10 == 0:  # 更频繁地打印进度
                print(f"已处理 {i * dataloader.batch_size} 张图像")
    
    features = torch.cat(features, 0)
    return features, file_paths

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整为更大的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载本地数据集
    dataset_path = r"D:\2025年实验照片"
    dataset = LocalImageDataset(root_dir=dataset_path, transform=transform)
    print(f"数据集根目录: {dataset_path}")
    
    if len(dataset) == 0:
        print("错误：未找到任何图像文件！请检查路径是否正确。")
        return
    
    # 使用较小的batch_size以适应更大的图像
    batch_size = 16  # 减小批次大小以适应更大的图像
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=2,
                          pin_memory=True)  # 启用pin_memory以加速数据传输
    
    # 创建模型实例
    model = CNNFeatureExtractor(num_classes=10).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("开始训练模型...")
    print(f"总图像数量: {len(dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"每轮训练批次数: {len(dataloader)}")
    
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=5)
    
    # 提取特征
    print("\n开始提取特征...")
    features, file_paths = extract_features(model, dataloader, device)
    
    # 保存特征和对应的文件路径
    output_dir = "extracted_features"
    os.makedirs(output_dir, exist_ok=True)
    
    features_np = features.numpy()
    np.save(os.path.join(output_dir, "features.npy"), features_np)
    
    # 保存文件路径列表
    with open(os.path.join(output_dir, "file_paths.txt"), "w", encoding="utf-8") as f:
        for path in file_paths:
            f.write(f"{path}\n")
    
    print(f"\n特征提取完成:")
    print(f"特征形状: {features.shape}")
    print(f"特征已保存到 {output_dir} 目录")
    print(f"处理的总图像数量: {len(dataset)}")

if __name__ == '__main__':
    main() 