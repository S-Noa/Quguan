# Linux服务器字体修复指南

## 问题现象

VGG训练日志中出现中文字体警告：
```
UserWarning: Glyph 39044 (\N{CJK UNIFIED IDEOGRAPH-9884}) missing from current font.
```

## 快速修复方案

### 方案1：自动安装脚本（推荐）

1. **给脚本执行权限**：
```bash
chmod +x install_chinese_fonts.sh
```

2. **运行安装脚本**：
```bash
./install_chinese_fonts.sh
```

3. **验证安装**：
```bash
python3 -c "from font_utils import CHINESE_SUPPORTED; print('中文支持:', CHINESE_SUPPORTED)"
```

### 方案2：手动安装（Ubuntu/Debian）

```bash
# 更新软件包列表
sudo apt-get update

# 安装文泉驿字体
sudo apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei

# 安装Google Noto CJK字体
sudo apt-get install -y fonts-noto-cjk

# 刷新字体缓存
fc-cache -fv
```

### 方案3：手动安装（CentOS/RHEL）

```bash
# 使用yum（老版本）
sudo yum install -y wqy-microhei-fonts wqy-zenhei-fonts
sudo yum install -y google-noto-cjk-fonts

# 或使用dnf（新版本）
sudo dnf install -y wqy-microhei-fonts wqy-zenhei-fonts
sudo dnf install -y google-noto-cjk-fonts

# 刷新字体缓存
fc-cache -fv
```

## 验证修复效果

### 1. 检查字体安装
```bash
# 检查文泉驿字体
fc-list | grep -i wqy

# 检查Noto字体
fc-list | grep -i noto

# 检查所有中文字体
fc-list :lang=zh
```

### 2. 测试Python字体支持
```bash
python3 -c "
from font_utils import CHINESE_SUPPORTED, get_labels
print('中文字体支持:', CHINESE_SUPPORTED)
if CHINESE_SUPPORTED:
    labels = get_labels(True)
    print('中文标签示例:', labels['true_concentration'])
else:
    labels = get_labels(False)
    print('英文标签示例:', labels['true_concentration'])
"
```

### 3. 生成测试图像
```bash
python3 -c "
import matplotlib.pyplot as plt
from font_utils import CHINESE_SUPPORTED, get_labels
import numpy as np

labels = get_labels(CHINESE_SUPPORTED)
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6)
plt.xlabel(labels['true_concentration'])
plt.ylabel(labels['predicted_concentration'])
plt.title('字体测试图')
plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
print('测试图像已保存: font_test.png')
"
```

## 重启训练程序

字体安装完成后，需要重启VGG训练程序：

1. **停止当前训练**（如果正在运行）：
```bash
# 找到训练进程
ps aux | grep vgg_regression

# 停止进程（替换PID）
kill -9 <PID>
```

2. **重新启动训练**：
```bash
nohup python3 vgg_regression.py --group allgroup > vgg_training.log 2>&1 &
```

3. **检查新日志**：
```bash
tail -f vgg_training.log
```

## 预期效果

修复成功后，您应该看到：

### 启动时的字体检测信息
```
检测到操作系统: Linux
系统可用字体数量: XXX
✓ 使用字体: WenQuanYi Micro Hei
```

### 无字体警告
训练过程中不再出现：
```
UserWarning: Glyph XXXXX missing from current font.
```

### 正确的图像标签
生成的预测图将显示清晰的中文标签而不是方框。

## 故障排除

### 如果安装失败
1. **检查网络连接**
2. **更新软件包管理器**：
   ```bash
   sudo apt-get update  # Ubuntu/Debian
   sudo yum update      # CentOS/RHEL
   ```
3. **检查权限**：确保有sudo权限

### 如果字体仍不生效
1. **重启Python程序**（必须）
2. **清理matplotlib缓存**：
   ```bash
   rm -rf ~/.cache/matplotlib
   ```
3. **检查字体路径**：
   ```bash
   fc-list | grep -i "wqy\|noto\|source"
   ```

## 备选方案

如果无法安装中文字体，修复后的代码会自动：
1. 使用英文标签
2. 抑制字体警告
3. 确保程序正常运行

这样即使没有中文字体，也不会影响训练过程。 