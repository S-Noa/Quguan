@echo off
title 云端模型优化训练启动器
color 0A

echo ===============================================
echo           云端模型优化训练启动器
echo ===============================================
echo.

:menu
echo 请选择训练模型类型:
echo.
echo 1. VGG+CBAM 优化训练 (bg0_20mw)
echo 2. VGG+CBAM 优化训练 (bg1_20mw)
echo 3. ResNet50 优化训练 (bg0_20mw)
echo 4. ResNet50 优化训练 (bg1_20mw)
echo 5. 自定义配置
echo 6. 退出
echo.

set /p choice=请输入选择 (1-6): 

if "%choice%"=="1" goto vgg_bg0
if "%choice%"=="2" goto vgg_bg1
if "%choice%"=="3" goto resnet_bg0
if "%choice%"=="4" goto resnet_bg1
if "%choice%"=="5" goto custom
if "%choice%"=="6" goto end
echo 无效选择，请重试...
goto menu

:vgg_bg0
echo.
echo 启动VGG+CBAM优化训练 (bg0_20mw)...
echo ===============================================
python enhanced_train_with_original_data.py --model_type vgg --bg_mode bg0_20mw
echo.
echo 训练完成！按任意键返回菜单...
pause >nul
goto menu

:vgg_bg1
echo.
echo 启动VGG+CBAM优化训练 (bg1_20mw)...
echo ===============================================
python enhanced_train_with_original_data.py --model_type vgg --bg_mode bg1_20mw
echo.
echo 训练完成！按任意键返回菜单...
pause >nul
goto menu

:resnet_bg0
echo.
echo 启动ResNet50优化训练 (bg0_20mw)...
echo ===============================================
python enhanced_train_with_original_data.py --model_type resnet50 --bg_mode bg0_20mw
echo.
echo 训练完成！按任意键返回菜单...
pause >nul
goto menu

:resnet_bg1
echo.
echo 启动ResNet50优化训练 (bg1_20mw)...
echo ===============================================
python enhanced_train_with_original_data.py --model_type resnet50 --bg_mode bg1_20mw
echo.
echo 训练完成！按任意键返回菜单...
pause >nul
goto menu

:custom
echo.
echo 自定义配置训练
echo ===============================================
echo.
set /p model_type=请输入模型类型 (vgg/resnet50): 
set /p bg_mode=请输入背景模式 (例如: bg0_20mw): 
set /p output_dir=请输入输出目录 (可选，直接回车使用默认): 

if "%output_dir%"=="" (
    python enhanced_train_with_original_data.py --model_type %model_type% --bg_mode %bg_mode%
) else (
    python enhanced_train_with_original_data.py --model_type %model_type% --bg_mode %bg_mode% --output_dir %output_dir%
)

echo.
echo 训练完成！按任意键返回菜单...
pause >nul
goto menu

:end
echo.
echo 再见！
pause 