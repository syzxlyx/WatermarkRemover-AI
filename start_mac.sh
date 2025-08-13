#!/usr/bin/env bash

# Mac M1 专用启动脚本
set -e

# 环境名称
ENV_NAME="py312aiwatermark"

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误：未找到conda，请先安装Miniconda"
    echo "下载地址：https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 检查环境是否存在
if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo "错误：环境 ${ENV_NAME} 不存在，请先运行 ./setup_mac.sh 进行安装"
    exit 1
fi

# 激活环境
echo "正在激活环境 ${ENV_NAME}..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 设置环境变量解决OpenMP冲突
export KMP_DUPLICATE_LIB_OK=TRUE

# 检查Python是否可用
if ! command -v python &> /dev/null; then
    echo "错误：Python命令不可用，请检查环境激活状态"
    exit 1
fi

echo "环境已激活，Python版本："
python --version

echo ""
echo "===================================="
echo "  WatermarkRemover-AI 启动"
echo "===================================="
echo ""
echo "选择启动方式："
echo "1. GUI界面 (推荐)"
echo "2. 命令行模式"
echo "3. 退出"
echo ""

read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        echo "启动GUI界面..."
        python remwmgui.py
        ;;
    2)
        echo "命令行模式"
        echo "使用方法：python remwm.py input_path output_path [options]"
        echo "示例：python remwm.py input.jpg output.jpg --transparent"
        echo ""
        echo "按 Ctrl+C 退出"
        exec bash
        ;;
    3)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择，启动GUI界面..."
        python remwmgui.py
        ;;
esac
