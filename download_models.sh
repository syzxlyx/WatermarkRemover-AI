#!/usr/bin/env bash

# IOPaint模型下载脚本
set -e

echo "======================================"
echo "  WatermarkRemover-AI 模型下载工具"
echo "======================================"

# 检查conda环境是否激活
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  请先激活conda环境："
    echo "   conda activate py312aiwatermark"
    exit 1
fi

echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 检查IOPaint是否安装
if ! python -c "import iopaint" 2>/dev/null; then
    echo "❌ IOPaint未安装，请先运行环境安装脚本"
    exit 1
fi

# 定义可用模型
declare -A MODELS
MODELS[lama]="LaMa - 主要水印移除模型（推荐）"
MODELS[fcf]="FCF - 快速卷积填充模型"
MODELS[mat]="MAT - 遮罩感知转换器"
MODELS[cv2]="CV2 - OpenCV快速填充（轻量级）"
MODELS[ldm]="LDM - 潜在扩散模型"
MODELS[zits]="ZITS - 增量变换器"
MODELS[wood]="WOOD - 水印检测优化"
MODELS[telea]="Telea - 快速修复算法"
MODELS[ns]="NS - Navier-Stokes填充"

# 函数：下载单个模型
download_model() {
    local model_name=$1
    local description=$2
    
    echo "正在下载: $description"
    echo "模型名称: $model_name"
    
    python -c "
try:
    from iopaint.model_manager import ModelManager
    print('开始下载模型...')
    ModelManager.download_model('$model_name')
    print('✅ $model_name 模型下载成功！')
except Exception as e:
    print(f'❌ 下载失败: {e}')
    print('请检查网络连接或稍后重试')
    exit(1)
" || {
        echo "❌ $model_name 模型下载失败"
        return 1
    }
    
    echo ""
}

# 函数：列出所有可用模型
list_models() {
    echo "可用模型列表："
    echo ""
    for model in "${!MODELS[@]}"; do
        echo "  $model - ${MODELS[$model]}"
    done
    echo ""
}

# 函数：下载推荐模型集合
download_recommended() {
    echo "下载推荐模型集合（lama, fcf, mat）..."
    echo ""
    
    recommended=("lama" "fcf" "mat")
    success_count=0
    
    for model in "${recommended[@]}"; do
        if download_model "$model" "${MODELS[$model]}"; then
            ((success_count++))
        fi
    done
    
    echo "======================================"
    echo "推荐模型下载完成：$success_count/3"
    echo "======================================"
}

# 函数：下载所有模型
download_all() {
    echo "下载所有可用模型..."
    echo "⚠️  这将需要较长时间和大量存储空间"
    echo ""
    
    read -p "确认继续？(y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
    
    success_count=0
    total_count=${#MODELS[@]}
    
    for model in "${!MODELS[@]}"; do
        if download_model "$model" "${MODELS[$model]}"; then
            ((success_count++))
        fi
    done
    
    echo "======================================"
    echo "模型下载完成：$success_count/$total_count"
    echo "======================================"
}

# 函数：检查已下载的模型
check_models() {
    echo "检查已下载的模型..."
    echo ""
    
    python -c "
try:
    from iopaint.model_manager import ModelManager
    available = ModelManager.available_models()
    print('已安装的模型：')
    if available:
        for model in available:
            print(f'  ✅ {model}')
    else:
        print('  ❌ 未找到已安装的模型')
except Exception as e:
    print(f'检查失败: {e}')
"
    echo ""
}

# 显示使用帮助
show_help() {
    echo "用法: $0 [选项] [模型名称]"
    echo ""
    echo "选项："
    echo "  --list, -l          列出所有可用模型"
    echo "  --recommended, -r   下载推荐模型集合"
    echo "  --all, -a          下载所有模型"
    echo "  --check, -c        检查已下载的模型"
    echo "  --help, -h         显示此帮助"
    echo ""
    echo "示例："
    echo "  $0 lama             下载LaMa模型"
    echo "  $0 --recommended    下载推荐模型"
    echo "  $0 --list          列出所有模型"
    echo ""
}

# 主程序
main() {
    case "${1:-}" in
        --list|-l)
            list_models
            ;;
        --recommended|-r)
            download_recommended
            ;;
        --all|-a)
            download_all
            ;;
        --check|-c)
            check_models
            ;;
        --help|-h|"")
            show_help
            ;;
        *)
            model_name="$1"
            if [[ -n "${MODELS[$model_name]:-}" ]]; then
                download_model "$model_name" "${MODELS[$model_name]}"
            else
                echo "❌ 未知模型：$model_name"
                echo ""
                list_models
                exit 1
            fi
            ;;
    esac
}

# 执行主程序
main "$@"
