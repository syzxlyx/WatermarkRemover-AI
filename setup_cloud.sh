#!/usr/bin/env bash
# 云环境专用安装脚本
# 解决scipy/numpy版本兼容性问题

set -e

echo "===================================="
echo "  WatermarkRemover-AI 云环境安装"
echo "===================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误：未找到conda，请先安装Miniconda"
    echo "下载地址：https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 环境名称
ENV_NAME="py312aiwatermark"

# 删除旧环境（如果存在）
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "发现已存在的环境，正在删除..."
    conda env remove -n ${ENV_NAME} -y
fi

# 检查是否有云环境配置文件
if [[ -f "environment_cloud.yml" ]]; then
    echo "使用云环境配置文件..."
    conda env create -f environment_cloud.yml
else
    echo "使用标准配置文件..."
    conda env create -f environment.yml
fi

# 激活环境
echo "正在激活环境..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE

# 修复scipy/numpy版本兼容性
echo "正在修复scipy/numpy版本兼容性..."
python fix_scipy_numpy.py

# 验证安装
echo "正在验证安装..."
python -c "
try:
    import numpy
    import scipy
    from scipy import integrate
    import torch
    import transformers
    from diffusers import schedulers
    from iopaint.model_manager import ModelManager
    from PyQt6 import QtCore
    import cv2
    print('✅ 所有核心依赖安装成功！')
    print(f'Python版本: {__import__('sys').version}')
    print(f'NumPy版本: {numpy.__version__}')
    print(f'SciPy版本: {scipy.__version__}')
    print(f'PyTorch版本: {torch.__version__}')
    print(f'Transformers版本: {transformers.__version__}')
except Exception as e:
    print(f'❌ 验证失败: {e}')
    print('请手动运行 python fix_scipy_numpy.py 修复')
    exit(1)
"

echo ""
echo "===================================="
echo "  云环境安装完成！"
echo "===================================="
echo ""
echo "使用方法："
echo "1. 激活环境：conda activate ${ENV_NAME}"
echo "2. 设置环境变量：export KMP_DUPLICATE_LIB_OK=TRUE"
echo "3. 启动程序："
echo "   - GUI模式：python remwmgui.py"
echo "   - 命令行：python remwm.py input.jpg output.jpg"
echo "   - 透明模式：python remwm.py input.jpg output.png --transparent"
echo ""
echo "注意事项："
echo "- 如果遇到scipy/numpy错误，运行：python fix_scipy_numpy.py"
echo "- 云环境建议使用透明模式以避免模型兼容性问题"
echo ""
