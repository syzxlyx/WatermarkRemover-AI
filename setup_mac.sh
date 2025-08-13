#!/usr/bin/env bash

# Mac M1 专用安装脚本
set -e

echo "===================================="
echo "  WatermarkRemover-AI Mac M1 安装"
echo "===================================="

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误：未找到conda，请先安装Miniconda"
    echo "下载地址：https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 检查是否为Mac M1
if [[ $(uname -m) != "arm64" ]]; then
    echo "警告：此脚本专为Mac M1 (ARM64) 设计"
fi

# 环境名称
ENV_NAME="py312aiwatermark"

# 删除旧环境（如果存在）
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "发现已存在的环境，正在删除..."
    conda env remove -n ${ENV_NAME} -y
fi

# 创建新环境
echo "正在创建conda环境..."
conda env create -f environment.yml

# 激活环境
echo "正在激活环境..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 设置环境变量解决OpenMP冲突
export KMP_DUPLICATE_LIB_OK=TRUE

# 所有依赖已在environment.yml中定义，conda会自动安装
echo "所有依赖已通过environment.yml安装完成"

# 验证安装
echo "正在验证安装..."
echo "检查核心依赖..."
python -c "
import torch
import transformers
import iopaint
from PyQt6 import QtCore
import cv2
print('✅ 所有核心依赖安装成功！')
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')
" || {
    echo "⚠️  部分依赖验证失败，但环境可能仍然可用"
    echo "可以尝试运行程序测试功能"
}

# 下载IOPaint模型
echo ""
echo "正在下载IOPaint模型..."
echo "这可能需要几分钟时间，请耐心等待..."

# 下载LaMa模型（主要的水印移除模型）
echo "下载LaMa模型..."
python -c "
try:
    from iopaint.model_manager import ModelManager
    ModelManager.download_model('lama')
    print('✅ LaMa模型下载成功')
except Exception as e:
    print(f'⚠️  LaMa模型下载失败: {e}')
    print('可以稍后手动下载: iopaint download --model lama')
" || echo "模型下载命令失败，可稍后手动执行"

# 可选：下载其他常用模型
echo "下载其他推荐模型（可选）..."
MODELS=("fcf" "mat" "cv2")
for model in "${MODELS[@]}"; do
    echo "尝试下载 $model 模型..."
    python -c "
try:
    from iopaint.model_manager import ModelManager
    ModelManager.download_model('$model')
    print('✅ $model 模型下载成功')
except Exception as e:
    print(f'⚠️  $model 模型下载失败: {e}')
" 2>/dev/null || echo "⚠️  $model 模型下载失败（跳过）"
done

echo ""
echo "===================================="
echo "  安装完成！"
echo "===================================="
echo ""
echo "使用方法："
echo "1. 激活环境：conda activate ${ENV_NAME}"
echo "2. 设置环境变量：export KMP_DUPLICATE_LIB_OK=TRUE"
echo "3. 启动GUI：python remwmgui.py"
echo "4. 命令行使用：python remwm.py input_path output_path"
echo ""
echo "注意：模型文件已预下载，如果使用其他模型，请运行："
echo "       iopaint download --model <model_name>"
echo ""

# 检查FFmpeg安装
echo "检查FFmpeg安装..."
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg已安装，支持视频音频处理"
    ffmpeg -version | head -1
else
    echo "⚠️  FFmpeg未安装，视频处理将无音频"
    echo "安装方法：brew install ffmpeg"
fi
echo ""
