#!/usr/bin/env python3
"""
Florence-2模型下载脚本
"""

import os
import sys
from pathlib import Path
import subprocess

def check_huggingface_login():
    """检查是否已登录Hugging Face"""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True, check=True)
        username = result.stdout.strip()
        if username:
            print(f"✅ 已登录Hugging Face，用户名: {username}")
            return True
        else:
            print("❌ 未登录Hugging Face")
            return False
    except subprocess.CalledProcessError:
        print("❌ 未登录Hugging Face")
        return False
    except FileNotFoundError:
        print("❌ 未找到huggingface-cli命令")
        return False

def login_huggingface():
    """登录Hugging Face"""
    print("🔐 请登录Hugging Face...")
    print("1. 访问 https://huggingface.co/settings/tokens")
    print("2. 创建新的Access Token")
    print("3. 复制Token")
    print("4. 运行以下命令登录:")
    print("   huggingface-cli login")
    print("   然后输入您的Token")
    
    choice = input("是否现在登录? (y/n): ").lower().strip()
    if choice == 'y':
        try:
            subprocess.run(['huggingface-cli', 'login'], check=True)
            print("✅ 登录成功！")
            return True
        except subprocess.CalledProcessError:
            print("❌ 登录失败")
            return False
    return False

def download_florence2_model():
    """下载Florence-2模型"""
    print("📥 开始下载Florence-2模型...")
    
    # 检查是否已登录
    if not check_huggingface_login():
        if not login_huggingface():
            print("❌ 无法登录Hugging Face，无法下载模型")
            return False
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        from huggingface_hub import snapshot_download
        
        # 创建模型目录
        model_dir = Path("./florence2_model")
        model_dir.mkdir(exist_ok=True)
        
        print("正在下载Florence-2模型...")
        print("这可能需要一些时间，请耐心等待...")
        
        # 下载模型
        snapshot_download(
            repo_id="microsoft/Florence-2-large",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )
        
        print("✅ Florence-2模型下载完成！")
        print(f"模型保存在: {model_dir.absolute()}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("💡 可能的解决方案:")
        print("1. 确保已登录Hugging Face: huggingface-cli login")
        print("2. 检查网络连接")
        print("3. 确保有足够的磁盘空间")
        print("4. 如果模型是私有的，需要申请访问权限")
        return False

def test_florence2_model():
    """测试Florence-2模型"""
    print("🧪 测试Florence-2模型...")
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        # 尝试加载模型
        model_path = "./florence2_model"
        if os.path.exists(model_path):
            print("从本地加载模型...")
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        else:
            print("从Hugging Face加载模型...")
            model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
            processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        
        print("✅ Florence-2模型加载成功！")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("  Florence-2模型下载工具")
    print("=" * 50)
    
    # 检查环境
    print("🔍 检查环境...")
    try:
        import torch
        import transformers
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ Transformers版本: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        print("请先激活conda环境: conda activate py312aiwatermark")
        return
    
    # 检查模型是否已存在
    model_dir = Path("./florence2_model")
    if model_dir.exists() and any(model_dir.iterdir()):
        print("📁 发现已存在的Florence-2模型")
        choice = input("是否重新下载? (y/n): ").lower().strip()
        if choice != 'y':
            print("使用现有模型...")
            if test_florence2_model():
                print("✅ 现有模型工作正常！")
                return
            else:
                print("现有模型有问题，需要重新下载...")
    
    # 下载模型
    if download_florence2_model():
        # 测试模型
        if test_florence2_model():
            print("🎉 Florence-2模型下载和测试完成！")
            print("现在您可以使用以下命令运行程序:")
            print("  python remwm.py input.jpg output.jpg")
        else:
            print("⚠️  模型下载完成但测试失败")
    else:
        print("❌ 模型下载失败")
        print("💡 建议使用修复版本: python remwm_fixed.py input.jpg output.jpg --transparent")

if __name__ == "__main__":
    main()
