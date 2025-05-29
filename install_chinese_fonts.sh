#!/bin/bash
# Linux中文字体安装脚本

echo "=== Linux中文字体安装脚本 ==="
echo "检测操作系统..."

# 检测Linux发行版
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
    OS=Debian
    VER=$(cat /etc/debian_version)
elif [ -f /etc/SuSe-release ]; then
    OS=openSUSE
elif [ -f /etc/redhat-release ]; then
    OS=RedHat
else
    OS=$(uname -s)
    VER=$(uname -r)
fi

echo "检测到系统: $OS $VER"

# 根据不同系统安装中文字体
case $OS in
    *Ubuntu*|*Debian*)
        echo "Ubuntu/Debian系统，安装中文字体..."
        sudo apt-get update
        echo "安装文泉驿字体..."
        sudo apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei
        echo "安装Google Noto CJK字体..."
        sudo apt-get install -y fonts-noto-cjk fonts-noto-cjk-extra
        echo "安装思源字体..."
        sudo apt-get install -y fonts-source-han-sans-cn fonts-source-han-serif-cn
        ;;
    *CentOS*|*Red\ Hat*|*RHEL*|*Fedora*)
        echo "RedHat系列系统，安装中文字体..."
        if command -v dnf &> /dev/null; then
            # Fedora/新版CentOS使用dnf
            echo "使用dnf安装..."
            sudo dnf install -y wqy-microhei-fonts wqy-zenhei-fonts
            sudo dnf install -y google-noto-cjk-fonts
            sudo dnf install -y adobe-source-han-sans-cn-fonts adobe-source-han-serif-cn-fonts
        else
            # 老版本CentOS使用yum
            echo "使用yum安装..."
            sudo yum install -y wqy-microhei-fonts wqy-zenhei-fonts
            sudo yum install -y google-noto-cjk-fonts
        fi
        ;;
    *openSUSE*)
        echo "openSUSE系统，安装中文字体..."
        sudo zypper install -y wqy-microhei-fonts wqy-zenhei-fonts
        sudo zypper install -y google-noto-cjk-fonts
        ;;
    *)
        echo "未识别的Linux发行版: $OS"
        echo "请手动安装中文字体包"
        echo "常见字体包名称："
        echo "  - fonts-wqy-microhei (文泉驿微米黑)"
        echo "  - fonts-wqy-zenhei (文泉驿正黑)"
        echo "  - fonts-noto-cjk (Google Noto CJK)"
        exit 1
        ;;
esac

echo ""
echo "=== 字体安装完成 ==="
echo "正在刷新字体缓存..."
fc-cache -fv

echo ""
echo "=== 验证字体安装 ==="
echo "检查已安装的中文字体："

# 检查字体是否安装成功
fonts_found=0

if fc-list | grep -i "wqy.*micro" > /dev/null; then
    echo "✓ 文泉驿微米黑 已安装"
    fonts_found=$((fonts_found + 1))
fi

if fc-list | grep -i "wqy.*zen" > /dev/null; then
    echo "✓ 文泉驿正黑 已安装"
    fonts_found=$((fonts_found + 1))
fi

if fc-list | grep -i "noto.*cjk" > /dev/null; then
    echo "✓ Google Noto CJK 已安装"
    fonts_found=$((fonts_found + 1))
fi

if fc-list | grep -i "source.*han" > /dev/null; then
    echo "✓ 思源字体 已安装"
    fonts_found=$((fonts_found + 1))
fi

echo ""
if [ $fonts_found -gt 0 ]; then
    echo "✅ 成功安装 $fonts_found 种中文字体"
    echo "现在可以重启Python程序以使用中文字体"
    echo ""
    echo "测试字体支持："
    echo "python3 -c \"from font_utils import CHINESE_SUPPORTED; print('中文支持:', CHINESE_SUPPORTED)\""
else
    echo "❌ 未检测到中文字体，安装可能失败"
    echo "请检查网络连接和软件包管理器配置"
fi

echo ""
echo "=== 安装脚本完成 ===" 