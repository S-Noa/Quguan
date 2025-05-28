import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from custom_dataset import LocalImageDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import os
from cbam import CBAM
import argparse

class VGGRegressionCBAM(nn.Module):
    def __init__(self, freeze_features=True):
        super(VGGRegressionCBAM, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential()
        # 在每个block后插入CBAM
        block_indices = [4, 9, 16, 23, 30]  # VGG16每个block的最后一层索引
        in_channels = [64, 128, 256, 512, 512]
        last_idx = 0
        for i, idx in enumerate(block_indices):
            for j in range(last_idx, idx+1):
                self.features.add_module(f'vgg_{j}', vgg.features[j])
            self.features.add_module(f'cbam_{i+1}', CBAM(in_channels[i]))
            last_idx = idx+1
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
        output = self.model(input_tensor)
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
    parser = argparse.ArgumentParser(description='VGG分组训练开关')
    parser.add_argument('--group', type=str, default='all', choices=['all', 'bg0', 'bg1', 'allgroup'],
                        help='选择训练数据分组: all(全部), bg0(不补光), bg1(补光), allgroup(全部分组)')
    args = parser.parse_args()

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

    if args.group in ['all', 'allgroup']:
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
        batch_size = 96
        num_workers = 8
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        model = VGGRegressionCBAM(freeze_features=True).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.reg_head.parameters(), lr=0.001)
        start_epoch = 0
        best_model_path = 'best_vgg_model.pth'
        if os.path.exists(best_model_path):
            print('检测到已有best_vgg_model.pth，正在加载权重并断点续训...')
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("\n开始训练模型...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, start_epoch=start_epoch, use_amp=True)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
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
        print("\n全部数据测试集结果:")
        print(f"均方误差 (MSE): {mse:.3f}")
        print(f"R² 分数: {r2:.3f}")
        plt.figure(figsize=(10, 6))
        plt.scatter(test_targets, test_predictions, alpha=0.5)
        plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
        plt.xlabel('真实浓度值')
        plt.ylabel('预测浓度值')
        plt.title(f'全部数据测试集预测结果 (R² = {r2:.3f})')
        plt.savefig('vgg_final_prediction_plot_all.png')
        plt.close()
        # Grad-CAM可视化
        print("生成全部数据模型的Grad-CAM可视化...")
        target_layer = None
        for name, module in model.features.named_modules():
            if 'cbam_5' in name:
                target_layer = module
        assert target_layer is not None, '未找到目标CBAM层'
        grad_cam = GradCAM(model, target_layer)
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
        cv2.imwrite('vgg_grad_cam_all.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print('Grad-CAM可视化结果已保存为 vgg_grad_cam_all.png')

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
            train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=8, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=8, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False, num_workers=8, pin_memory=True)
            model = VGGRegressionCBAM(freeze_features=True).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.reg_head.parameters(), lr=0.001)
            best_model_path = f'best_vgg_model_{bg_type}.pth'
            print(f"\n开始训练{bg_type}模型...")
            train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, start_epoch=0, use_amp=True)
            model.load_state_dict(torch.load(best_model_path, map_location=device))
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
            print(f"\n{bg_type}测试集结果:")
            print(f"均方误差 (MSE): {mse:.3f}")
            print(f"R² 分数: {r2:.3f}")
            plt.figure(figsize=(10, 6))
            plt.scatter(test_targets, test_predictions, alpha=0.5)
            plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
            plt.xlabel('真实浓度值')
            plt.ylabel('预测浓度值')
            plt.title(f'{bg_type}测试集预测结果 (R² = {r2:.3f})')
            plt.savefig(f'vgg_final_prediction_plot_{bg_type}.png')
            plt.close()
            # Grad-CAM可视化
            print(f"生成{bg_type}模型的Grad-CAM可视化...")
            target_layer = None
            for name, module in model.features.named_modules():
                if 'cbam_5' in name:
                    target_layer = module
            assert target_layer is not None, '未找到目标CBAM层'
            grad_cam = GradCAM(model, target_layer)
            img, _ = dataset[0]
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
            cv2.imwrite(f'vgg_grad_cam_{bg_type}.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f'Grad-CAM可视化结果已保存为 vgg_grad_cam_{bg_type}.png')

if __name__ == '__main__':
    main()

# Grad-CAM可视化调用示例
if __name__ == '__main__':
    # 只在测试集上可视化一张图片
    import cv2
    from torchvision.utils import make_grid
    import matplotlib.cm as cm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载训练好的模型
    model = VGGRegressionCBAM(freeze_features=True).to(device)
    model.load_state_dict(torch.load('best_vgg_model.pth', map_location=device))
    model.eval()

    # 选择要可视化的层（最后一个CBAM）
    target_layer = None
    for name, module in model.features.named_modules():
        if 'cbam_5' in name:
            target_layer = module
    assert target_layer is not None, '未找到目标CBAM层'

    grad_cam = GradCAM(model, target_layer)

    # 随机选一张测试集图片
    from custom_dataset import LocalImageDataset
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_path = r"D:\2025年实验照片"
    test_dataset = LocalImageDataset(root_dir=dataset_path, transform=transform)
    # 取第一张图片
    img, _ = test_dataset[0]
    input_tensor = img.unsqueeze(0).to(device)

    # 生成Grad-CAM热力图
    cam = grad_cam(input_tensor)
    # 反归一化还原原图
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    # 调整热力图尺寸
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cm.jet(cam_resized)[:, :, :3]  # 取RGB
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    # 保存和显示
    cv2.imwrite('grad_cam_cbam_vgg.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print('Grad-CAM可视化结果已保存为 grad_cam_cbam_vgg.png') 