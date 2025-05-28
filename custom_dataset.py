import os
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LocalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, bg_type=None):
        """
        参数:
            root_dir (string): 图像文件夹的根目录
            transform (callable, optional): 可选的图像转换
            bg_type (str, optional): 'bg0'或'bg1'，只加载对应补光类型的图片
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.concentrations = []
        
        # 支持的图像格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        # 用于提取浓度值的正则表达式
        concentration_pattern = r'\d+°-(\d+)-'
        
        # 递归遍历目录获取所有图像文件
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    if bg_type is not None and bg_type not in file:
                        continue  # 跳过不符合补光类型的图片
                    file_path = os.path.join(root, file)
                    # 尝试从文件名中提取浓度值
                    match = re.search(concentration_pattern, file)
                    if match:
                        concentration = float(match.group(1))
                        self.image_files.append(file_path)
                        self.concentrations.append(concentration)
        
        # 数据统计
        if self.concentrations:
            min_conc = min(self.concentrations)
            max_conc = max(self.concentrations)
            avg_conc = sum(self.concentrations) / len(self.concentrations)
            print(f"找到 {len(self.image_files)} 张有效图像 (bg_type={bg_type})")
            print(f"浓度范围: {min_conc} - {max_conc}, 平均值: {avg_conc:.2f}")
        else:
            print(f"警告：未找到任何有效的图像文件 (bg_type={bg_type})！")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        concentration = self.concentrations[idx]
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            # 如果图像损坏，返回一个空白图像
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和浓度值
        return image, concentration 