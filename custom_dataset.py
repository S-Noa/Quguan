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
        
        # 临时存储所有候选文件
        candidate_files = []
        candidate_concentrations = []
        
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
                        candidate_files.append(file_path)
                        candidate_concentrations.append(concentration)
        
        print(f"找到 {len(candidate_files)} 个候选图像文件 (bg_type={bg_type})")
        
        # 验证图像文件完整性，过滤损坏文件
        print("正在验证图像文件完整性...")
        corrupted_count = 0
        
        for i, (file_path, concentration) in enumerate(zip(candidate_files, candidate_concentrations)):
            try:
                # 快速验证图像文件
                with Image.open(file_path) as img:
                    img.verify()
                # 添加到有效文件列表
                self.image_files.append(file_path)
                self.concentrations.append(concentration)
            except (OSError, IOError, Image.UnidentifiedImageError) as e:
                corrupted_count += 1
                print(f"跳过损坏图像: {file_path} - {e}")
        
        # 数据统计
        if self.concentrations:
            min_conc = min(self.concentrations)
            max_conc = max(self.concentrations)
            avg_conc = sum(self.concentrations) / len(self.concentrations)
            print(f"验证完成！有效图像: {len(self.image_files)} 张 (bg_type={bg_type})")
            print(f"损坏图像: {corrupted_count} 张 (已跳过)")
            print(f"损坏率: {corrupted_count/(len(self.image_files)+corrupted_count)*100:.2f}%")
            print(f"浓度范围: {min_conc} - {max_conc}, 平均值: {avg_conc:.2f}")
        else:
            print(f"警告：未找到任何有效的图像文件 (bg_type={bg_type})！")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        concentration = self.concentrations[idx]
        
        # 由于已经预先验证过，这里可以简化错误处理
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 理论上不应该到达这里，但保留作为最后防线
            print(f"意外错误：无法读取已验证的图像 {img_path}: {e}")
            # 创建一个临时图像作为应急措施
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和浓度值
        return image, concentration 