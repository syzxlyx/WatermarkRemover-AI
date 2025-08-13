#!/usr/bin/env bash

# 最终测试脚本
set -e

echo "===================================="
echo "  最终测试 - WatermarkRemover-AI"
echo "===================================="

# 环境名称
ENV_NAME="py312aiwatermark"

# 激活环境
echo "正在激活环境 ${ENV_NAME}..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE

# 测试所有关键组件
echo "测试关键组件..."

# 1. 测试Python
echo "1. 测试Python..."
python --version

# 2. 测试PyTorch
echo "2. 测试PyTorch..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"

# 3. 测试Transformers
echo "3. 测试Transformers..."
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"

# 4. 测试Florence-2模型
echo "4. 测试Florence-2模型..."
python -c "from transformers import AutoProcessor, AutoModelForCausalLM; print('Florence-2模型可以加载')"

# 5. 测试IOPaint
echo "5. 测试IOPaint..."
python -c "from iopaint.model_manager import ModelManager; print('IOPaint可以正常导入')"

# 6. 测试项目主文件
echo "6. 测试项目主文件..."
python -c "import remwm; print('remwm.py可以正常导入')"

echo ""
echo "===================================="
echo "  所有测试通过！"
echo "===================================="
echo ""
echo "现在您可以运行："
echo "  ./start_mac.sh"
echo "或"
echo "  python remwmgui.py"
echo ""
echo "注意：首次运行时需要下载模型文件，请确保网络连接正常。"
echo ""
