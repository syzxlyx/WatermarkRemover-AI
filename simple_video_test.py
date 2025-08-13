#!/usr/bin/env python3
"""
简单的视频处理测试脚本
使用FFmpeg方法避免OpenCV VideoWriter问题
"""

import subprocess
import tempfile
import shutil
import os
from pathlib import Path
import sys
from PIL import Image, ImageDraw
import numpy as np

def create_test_watermark_removal(input_path, output_path):
    """创建一个简单的水印移除测试"""
    print(f"🎬 开始简单视频处理测试: {input_path}")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    frames_dir = Path(temp_dir) / "frames"
    frames_dir.mkdir()
    
    try:
        # 步骤1: 提取前10帧进行测试
        print("📤 提取测试帧...")
        extract_cmd = [
            "ffmpeg", "-i", str(input_path),
            "-vf", "select='lt(n,10)'",  # 只提取前10帧
            "-vsync", "0",
            str(frames_dir / "frame_%03d.png")
        ]
        
        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 提取帧失败: {result.stderr}")
            return False
        
        # 检查提取的帧
        frame_files = list(frames_dir.glob("frame_*.png"))
        print(f"📊 提取了 {len(frame_files)} 帧")
        
        if len(frame_files) == 0:
            print("❌ 没有提取到帧")
            return False
        
        # 步骤2: 处理每一帧（添加简单的水印移除效果）
        print("🔄 处理帧...")
        for frame_file in frame_files:
            try:
                # 读取图像
                image = Image.open(frame_file)
                width, height = image.size
                
                # 在右下角添加一个白色矩形（模拟水印移除）
                draw = ImageDraw.Draw(image)
                rect_size = min(width, height) // 10
                x1 = width - rect_size - 20
                y1 = height - rect_size - 20
                x2 = width - 20
                y2 = height - 20
                
                draw.rectangle([x1, y1, x2, y2], fill='white', outline='red', width=2)
                
                # 保存处理后的帧
                image.save(frame_file)
                print(f"✅ 处理完成: {frame_file.name}")
                
            except Exception as e:
                print(f"⚠️  处理帧 {frame_file} 失败: {e}")
        
        # 步骤3: 重新组合为视频
        print("🔧 重新组合视频...")
        temp_video = Path(temp_dir) / "test_video.mp4"
        
        # 获取原视频的帧率
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
            str(input_path)
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            fps_str = result.stdout.strip()
            # 处理分数形式的帧率
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
        else:
            fps = 30.0  # 默认帧率
        
        print(f"📺 使用帧率: {fps} FPS")
        
        compose_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%03d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-t", "1",  # 只生成1秒的视频
            str(temp_video)
        ]
        
        result = subprocess.run(compose_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 视频组合失败: {result.stderr}")
            return False
        
        # 步骤4: 添加原始音频
        print("🔊 添加音频...")
        final_cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video),      # 处理后的视频
            "-i", str(input_path),      # 原始视频（含音频）
            "-c:v", "copy",             # 复制视频流
            "-c:a", "aac",              # 重新编码音频
            "-map", "0:v:0",            # 使用第一个文件的视频
            "-map", "1:a:0",            # 使用第二个文件的音频
            "-t", "1",                  # 只输出1秒
            str(output_path)
        ]
        
        result = subprocess.run(final_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 音频合并失败: {result.stderr}")
            # 如果音频合并失败，直接复制无音频版本
            shutil.copy2(temp_video, output_path)
            print("📋 已复制无音频版本")
        else:
            print("✅ 音频合并成功!")
        
        # 检查输出文件
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"📁 输出文件大小: {size} bytes")
            
            # 验证输出文件
            verify_cmd = ["ffprobe", "-v", "quiet", "-show_streams", str(output_path)]
            result = subprocess.run(verify_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ 输出文件验证成功")
                return True
            else:
                print("❌ 输出文件验证失败")
                return False
        else:
            print("❌ 输出文件未生成")
            return False
            
    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def main():
    if len(sys.argv) != 3:
        print("用法: python simple_video_test.py <input_video> <output_video>")
        print("这个脚本会处理输入视频的前1秒作为测试")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    
    print("🧪 简单视频处理测试工具")
    print("=" * 50)
    print("这个工具会:")
    print("1. 提取输入视频的前10帧")
    print("2. 在每帧右下角添加白色矩形（模拟水印移除）")
    print("3. 重新组合为1秒的测试视频")
    print("4. 添加原始音频")
    print()
    
    success = create_test_watermark_removal(input_path, output_path)
    
    if success:
        print(f"🎉 测试完成! 输出文件: {output_path}")
        print("请检查输出视频是否同时包含图像和声音")
    else:
        print("❌ 测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
