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
import argparse

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, start_epoch=0, best_model_path='best_model.pth', group_tag=None):
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
            torch.save(model.state_dict(), best_model_path)
            print(f'保存最佳模型，验证损失: {val_loss:.3f}')
        if epoch % 5 == 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(val_targets, val_predictions, alpha=0.5)
            plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], 'r--')
            plt.xlabel('真实浓度值')
            plt.ylabel('预测浓度值')
            plt.title(f'Epoch {epoch + 1} 预测值 vs 真实值 (R² = {r2:.3f})')
            tag = f'_{group_tag}' if group_tag else ''
            plt.savefig(f'cnn_prediction_plot_epoch_{epoch + 1}{tag}.png')
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

# Grad-CAM可视化工具
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        handle1 = self.target_layer.register_forward_hook(forward_hook)
        handle2 = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle1, handle2])
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    def __call__(self, input_tensor, target_index=None):
        self.model.eval()
        output, _ = self.model(input_tensor)
        if target_index is None:
            target_index = 0
        loss = output[:, 0].sum() if output.ndim == 2 else output.sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        self.remove_hooks()
        return cam

def main():
    parser = argparse.ArgumentParser(description='CNN分组训练开关')
    parser.add_argument('--group', type=str, default='all', choices=['all', 'bg0', 'bg1', 'allgroup'],
                        help='选择训练数据分组: all(全部), bg0(不补光), bg1(补光), allgroup(全部分组)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    dataset_path = r"D:\2025年实验照片"

    if args.group in ['all', 'allgroup']:
        # 全部数据训练
        print("\n【全部数据训练】")
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
        batch_size = 16
        num_workers = 2
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        model = CNNFeatureExtractor().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        start_epoch = 2
        if os.path.exists('best_model.pth'):
            print('检测到已有best_model.pth，正在加载权重并断点续训...')
            model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("\n开始训练模型...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15, start_epoch=start_epoch, group_tag='all')
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        mse, r2, predictions, targets = evaluate_model(model, test_loader, device)
        print("\n全部数据测试集结果:")
        print(f"均方误差 (MSE): {mse:.3f}")
        print(f"R² 分数: {r2:.3f}")
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.xlabel('真实浓度值')
        plt.ylabel('预测浓度值')
        plt.title(f'全部数据测试集预测结果 (R² = {r2:.3f})')
        plt.savefig('cnn_final_prediction_plot_all.png')
        plt.close()
        # Grad-CAM可视化（全部数据）
        print("生成全部数据模型的Grad-CAM可视化...")
        grad_cam = GradCAM(model, model.last_conv)
        img, _ = full_dataset[0]
        input_tensor = img.unsqueeze(0).to(device)
        cam = grad_cam(input_tensor)
        import cv2
        import matplotlib.cm as cm
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        cv2.imwrite('cnn_grad_cam_all.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print('Grad-CAM可视化结果已保存为 cnn_grad_cam_all.png')

    if args.group in ['bg0', 'bg1', 'allgroup']:
        group_list = [args.group] if args.group in ['bg0', 'bg1'] else ['bg0', 'bg1']
        for bg_type in group_list:
            print(f"\n【仅{bg_type}图像训练】")
            dataset = LocalImageDataset(root_dir=dataset_path, transform=transform, bg_type=bg_type)
            if len(dataset) < 10:
                print(f"{bg_type}样本过少，跳过...")
                continue
            train_size = int(0.7 * len(dataset))
            val_size = int(0.15 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            print(f"{bg_type}数据集划分:")
            print(f"训练集: {len(train_dataset)} 张图像")
            print(f"验证集: {len(val_dataset)} 张图像")
            print(f"测试集: {len(test_dataset)} 张图像")
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
            model = CNNFeatureExtractor().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            print(f"\n开始训练{bg_type}模型...")
            best_model_path = f'best_model_{bg_type}.pth'
            start_epoch = 1
            if os.path.exists(best_model_path):
                print(f'检测到已有{best_model_path}，正在加载权重并断点续训...')
                model.load_state_dict(torch.load(best_model_path, map_location=device))
            train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15, start_epoch=start_epoch, best_model_path=best_model_path, group_tag=bg_type)
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            mse, r2, predictions, targets = evaluate_model(model, test_loader, device)
            print(f"\n{bg_type}测试集结果:")
            print(f"均方误差 (MSE): {mse:.3f}")
            print(f"R² 分数: {r2:.3f}")
            plt.figure(figsize=(10, 6))
            plt.scatter(targets, predictions, alpha=0.5)
            plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
            plt.xlabel('真实浓度值')
            plt.ylabel('预测浓度值')
            plt.title(f'{bg_type}测试集预测结果 (R² = {r2:.3f})')
            plt.savefig(f'cnn_final_prediction_plot_{bg_type}.png')
            plt.close()
            # Grad-CAM可视化（分组）
            print(f"生成{bg_type}模型的Grad-CAM可视化...")
            grad_cam = GradCAM(model, model.last_conv)
            img, _ = dataset[0]
            input_tensor = img.unsqueeze(0).to(device)
            cam = grad_cam(input_tensor)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img_np = np.clip(img_np, 0, 1)
            img_np = (img_np * 255).astype(np.uint8)
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = cm.jet(cam_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
            cv2.imwrite(f'cnn_grad_cam_{bg_type}.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f'Grad-CAM可视化结果已保存为 cnn_grad_cam_{bg_type}.png')

if __name__ == '__main__':
    main() 