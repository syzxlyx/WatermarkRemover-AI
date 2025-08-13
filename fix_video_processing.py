#!/usr/bin/env python3
"""
修复版本的视频处理脚本
专门解决"只有声音没有图像"的问题
"""

import cv2
import numpy as np
from PIL import Image
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
import sys

def create_simple_mask(image, mask_percent=5.0):
    """创建简单的水印掩码（用于测试）"""
    width, height = image.size
    mask = Image.new('L', (width, height), 0)
    
    # 在右下角创建一个小矩形掩码
    mask_width = int(width * mask_percent / 100)
    mask_height = int(height * mask_percent / 100)
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    x1 = width - mask_width - 10
    y1 = height - mask_height - 10
    x2 = width - 10
    y2 = height - 10
    
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask

def simple_inpaint(image_array, mask_array):
    """简单的图像修复（用白色填充）"""
    result = image_array.copy()
    mask_indices = mask_array > 0
    result[mask_indices] = [255, 255, 255]  # 白色填充
    return result

def process_video_fixed(input_path, output_path):
    """修复版本的视频处理函数 - 使用FFmpeg方式"""
    print(f"🎬 开始处理视频: {input_path}")

    # 打开输入视频获取基本信息
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {input_path}")
        return False

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📺 视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    frames_dir = Path(temp_dir) / "frames"
    frames_dir.mkdir()

    try:
        # 方法1: 提取帧 -> 处理 -> 重新组合
        print("🔄 提取视频帧...")

        # 使用FFmpeg提取帧
        extract_cmd = [
            "ffmpeg", "-i", str(input_path),
            "-vf", f"fps={fps}",  # 保持原始帧率
            str(frames_dir / "frame_%06d.png")
        ]

        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 提取帧失败: {result.stderr}")
            return False

        # 检查提取的帧数
        frame_files = list(frames_dir.glob("frame_*.png"))
        print(f"📊 成功提取 {len(frame_files)} 帧")

        if len(frame_files) == 0:
            print("❌ 没有提取到任何帧")
            return False

        # 处理每一帧
        print("🔄 处理帧...")
        processed_count = 0

        for i, frame_file in enumerate(sorted(frame_files)):
            try:
                # 读取帧
                pil_image = Image.open(frame_file).convert('RGB')

                # 创建简单掩码
                mask_image = create_simple_mask(pil_image)

                # 简单修复
                result_array = simple_inpaint(np.array(pil_image), np.array(mask_image))
                result_image = Image.fromarray(result_array)

                # 保存处理后的帧
                result_image.save(frame_file)
                processed_count += 1

                # 显示进度
                if (i + 1) % 30 == 0:
                    progress = ((i + 1) / len(frame_files)) * 100
                    print(f"📊 处理进度: {i + 1}/{len(frame_files)} ({progress:.1f}%)")

            except Exception as e:
                print(f"⚠️  处理帧 {frame_file} 时出错: {e}")

        print(f"✅ 帧处理完成: {processed_count}/{len(frame_files)} 帧成功处理")

        # 使用FFmpeg重新组合视频（无音频）
        temp_video_path = Path(temp_dir) / "temp_video.mp4"

        compose_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            str(temp_video_path)
        ]

        print("🔧 重新组合视频...")
        result = subprocess.run(compose_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 视频组合失败: {result.stderr}")
            return False

        # 检查临时视频文件
        if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
            print("❌ 临时视频文件生成失败")
            return False

        print(f"📁 临时视频大小: {os.path.getsize(temp_video_path)} bytes")

        # 合并音视频
        return merge_audio_video(input_path, temp_video_path, output_path)

    finally:
        # 清理临时文件
        cap.release()
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

def merge_audio_video(original_video, processed_video, output_path):
    """合并音视频"""
    print("🔊 开始合并音视频...")
    
    try:
        # 检查FFmpeg是否可用
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        
        # 使用FFmpeg合并
        cmd = [
            "ffmpeg", "-y",
            "-i", str(processed_video),  # 处理后的视频
            "-i", str(original_video),   # 原始视频（含音频）
            "-c:v", "copy",              # 复制视频流
            "-c:a", "aac",               # 重新编码音频
            "-map", "0:v:0",             # 使用第一个文件的视频
            "-map", "1:a:0",             # 使用第二个文件的音频
            "-shortest",                 # 以较短的流为准
            str(output_path)
        ]
        
        print(f"🔧 执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 音视频合并成功!")
            return True
        else:
            print(f"❌ FFmpeg合并失败: {result.stderr}")
            # 如果合并失败，直接复制处理后的视频（无音频）
            print("📋 复制无音频视频...")
            shutil.copy2(processed_video, output_path)
            return True
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpeg不可用，复制无音频视频...")
        shutil.copy2(processed_video, output_path)
        return True

def main():
    if len(sys.argv) != 3:
        print("用法: python fix_video_processing.py <input_video> <output_video>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    
    print("🚀 修复版视频处理工具")
    print("=" * 50)
    
    success = process_video_fixed(input_path, output_path)
    
    if success:
        print(f"🎉 处理完成! 输出文件: {output_path}")
    else:
        print("❌ 处理失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
