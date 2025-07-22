#!/usr/bin/env python
# -*- coding: utf-8 -*-"""
#九宫格图像生成工具
#支持将指定目录下的图像文件排列成4宫格图、9宫格图或16宫格图并保存

#使用方法:
#python grid_image_generator.py --input_dir ./images --grid_size 9 --output_path grid_result.jpg
import os
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union

class GridImageGenerator:
    def __init__(self) -> None:
        """初始化九宫格图像生成器"""
        # 支持的图像格式
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    def get_image_paths(self, input_dir: str) -> List[str]:
        """
        获取目录下所有支持的图像文件路径
        :param input_dir: 输入目录路径
        :return: 图像文件路径列表
        """
        if not os.path.isdir(input_dir):
            raise ValueError(f"输入目录不存在: {input_dir}")

        image_paths = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(self.supported_formats):
                image_paths.append(os.path.join(input_dir, filename))

        if not image_paths:
            raise ValueError(f"在目录 {input_dir} 中未找到支持的图像文件")

        return sorted(image_paths)

    def load_and_preprocess_image(self, image_path: str, target_size: Tuple[int, int], padding_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """
        加载并预处理图像
        :param image_path: 图像路径
        :param target_size: 目标尺寸 (width, height)
        :param padding_color: 填充颜色 (R, G, B)
        :return: 预处理后的图像
        """
        try:
            with Image.open(image_path) as img:
                # 转换为RGB模式以统一格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 计算调整大小以保持纵横比
                img.thumbnail(target_size)
                img_width, img_height = img.size

                # 创建空白画布并将图像居中放置
                canvas = Image.new('RGB', target_size, padding_color)
                offset_x = (target_size[0] - img_width) // 2
                offset_y = (target_size[1] - img_height) // 2
                canvas.paste(img, (offset_x, offset_y))

                return canvas
        except Exception as e:
            print(f"⚠️ 警告: 无法加载图像 {image_path}, 错误: {str(e)}")
            # 返回空白图像
            return Image.new('RGB', target_size, padding_color)

    def create_grid_image(self,
                         image_paths: List[str],
                         grid_size: int = 9,
                         output_path: str = 'grid_result.jpg',
                         cell_size: Tuple[int, int] = (200, 200),
                         padding: int = 10,
                         padding_color: Tuple[int, int, int] = (255, 255, 255),
                         show_filenames: bool = True,
                         filename_font_size: int = 12) -> None:
        """
        创建九宫格图像
        :param image_paths: 图像文件路径列表
        :param grid_size: 宫格数量，支持4, 9, 16
        :param output_path: 输出图像路径
        :param cell_size: 每个单元格的尺寸 (width, height)
        :param padding: 单元格之间的间距
        :param padding_color: 背景和间距颜色
        :param show_filenames: 是否显示文件名
        :param filename_font_size: 文件名字体大小
        """
        # 验证宫格大小
        if grid_size not in [4, 9, 16]:
            raise ValueError(f"不支持的宫格大小 {grid_size}, 请选择4, 9或16")

        # 计算行列数
        cols = int(np.sqrt(grid_size))
        rows = cols

        # 限制图像数量不超过宫格数
        num_images = min(len(image_paths), grid_size)
        image_paths = image_paths[:num_images]

        # 计算画布大小
        canvas_width = cols * cell_size[0] + (cols - 1) * padding
        canvas_height = rows * cell_size[1] + (rows - 1) * padding

        # 如果显示文件名，增加底部空间
        if show_filenames:
            canvas_height += filename_font_size + 10  # 额外空间用于文件名

        # 创建画布
        grid_image = Image.new('RGB', (canvas_width, canvas_height), padding_color)
        draw = ImageDraw.Draw(grid_image) if show_filenames else None

        # 放置图像
        for i, image_path in enumerate(image_paths):
            row = i // cols
            col = i % cols

            # 计算位置
            x = col * (cell_size[0] + padding)
            y = row * (cell_size[1] + padding)

            # 加载并预处理图像
            img = self.load_and_preprocess_image(image_path, cell_size, padding_color)

            # 粘贴到画布
            grid_image.paste(img, (x, y))

            # 绘制文件名
            if show_filenames and draw:
                filename = os.path.splitext(os.path.basename(image_path))[0]
                text_y = y + cell_size[1] + 5
                draw.text((x, text_y), filename, fill=(0, 0, 0))

        # 保存结果
        grid_image.save(output_path)
        print(f"✅ 九宫格图像已保存至: {output_path}")

    def run(self) -> None:
        """运行九宫格图像生成器"""
        parser = argparse.ArgumentParser(description='九宫格图像生成工具', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--input_dir', type=str, required=True, help='包含图像的目录路径')
        parser.add_argument('--grid_size', type=int, default=9, choices=[4, 9, 16], help='宫格数量: 4(2x2), 9(3x3), 16(4x4)')
        parser.add_argument('--output_path', type=str, default='grid_result.jpg', help='输出图像路径')
        parser.add_argument('--cell_size', type=int, default=200, help='每个单元格的尺寸 (像素)')
        parser.add_argument('--padding', type=int, default=10, help='单元格之间的间距 (像素)')
        parser.add_argument('--padding_color', type=str, default='255,255,255', help='背景和间距颜色 (R,G,B)')
        parser.add_argument('--no_filenames', action='store_true', help='不显示文件名')

        args = parser.parse_args()

        try:
            # 解析颜色参数
            padding_color = tuple(map(int, args.padding_color.split(',')))
            if len(padding_color) != 3 or any(c < 0 or c > 255 for c in padding_color):
                raise ValueError("颜色值必须是0-255之间的三个整数")
        except:
            print("⚠️ 警告: 颜色格式无效，使用默认白色背景")
            padding_color = (255, 255, 255)

        try:
            print("🔍 正在查找图像文件...")
            image_paths = self.get_image_paths(args.input_dir)
            print(f"📁 找到 {len(image_paths)} 个图像文件")

            print("🖼️ 正在创建九宫格图像...")
            self.create_grid_image(
                image_paths,
                grid_size=args.grid_size,
                output_path=args.output_path,
                cell_size=(args.cell_size, args.cell_size),
                padding=args.padding,
                padding_color=padding_color,
                show_filenames=not args.no_filenames
            )

        except Exception as e:
            print(f"❌ 生成九宫格图像失败: {str(e)}")
            exit(1)

if __name__ == '__main__':
    generator = GridImageGenerator()
    generator.run()