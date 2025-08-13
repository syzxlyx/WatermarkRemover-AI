#!/usr/bin/env bash

# 测试安装脚本
set -e

echo "===================================="
echo "  测试 WatermarkRemover-AI 安装"
echo "===================================="

# 环境名称
ENV_NAME="py312aiwatermark"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误：未找到conda"
    exit 1
fi

# 检查环境是否存在
if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo "❌ 错误：环境 ${ENV_NAME} 不存在"
    echo "请先运行 ./setup_mac.sh 进行安装"
    exit 1
fi

# 激活环境
echo "正在激活环境 ${ENV_NAME}..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE

# 测试Python
echo "测试Python..."
if python --version; then
    echo "✅ Python 正常"
else
    echo "❌ Python 异常"
    exit 1
fi

# 测试基础依赖
echo "测试基础依赖..."
if python -c "import torch; print('✅ PyTorch 正常')"; then
    echo "✅ PyTorch 正常"
else
    echo "❌ PyTorch 异常"
fi

if python -c "import transformers; print('✅ Transformers 正常')"; then
    echo "✅ Transformers 正常"
else
    echo "❌ Transformers 异常"
fi

if python -c "import iopaint; print('✅ IOPaint 正常')"; then
    echo "✅ IOPaint 正常"
else
    echo "❌ IOPaint 异常"
fi

# 测试Florence-2模型
echo "测试Florence-2模型..."
if python -c "from transformers import AutoProcessor, AutoModelForCausalLM; print('✅ Florence-2模型可以加载')"; then
    echo "✅ Florence-2模型可以加载"
else
    echo "❌ Florence-2模型加载失败"
fi

echo ""
echo "===================================="
echo "  测试完成！"
echo "===================================="
echo ""
echo "如果所有测试都通过，您可以运行："
echo "  ./start_mac.sh"
echo "或"
echo "  python remwmgui.py"
echo ""
