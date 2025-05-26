import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LocalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        参数:
            root_dir (string): 图像文件夹的根目录
            transform (callable, optional): 可选的图像转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # 支持的图像格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        # 递归遍历目录获取所有图像文件
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.image_files.append(os.path.join(root, file))
        
        print(f"找到 {len(self.image_files)} 张图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法读取图像 {img_path}: {e}")
            # 如果图像损坏，返回一个空白图像
            image = Image.new('RGB', (64, 64), color='black')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 由于这是特征提取任务，我们不需要标签，返回0作为虚拟标签
        return image, 0 