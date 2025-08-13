#!/usr/bin/env python3
"""
视频处理调试脚本
用于诊断视频转换过程中的问题
"""

import cv2
import subprocess
import sys
import os
from pathlib import Path

def check_video_info(video_path):
    """检查视频文件信息"""
    print(f"\n=== 检查视频文件: {video_path} ===")
    
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        return False
    
    file_size = os.path.getsize(video_path)
    print(f"📁 文件大小: {file_size} bytes")
    
    # 使用OpenCV检查
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ OpenCV无法打开视频文件")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📺 分辨率: {width}x{height}")
    print(f"🎬 帧率: {fps} FPS")
    print(f"📊 总帧数: {total_frames}")
    
    # 尝试读取第一帧
    ret, frame = cap.read()
    if ret:
        print("✅ 可以读取视频帧")
    else:
        print("❌ 无法读取视频帧")
    
    cap.release()
    
    # 使用FFprobe检查（如果可用）
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFprobe可以读取文件")
            # 检查是否有视频流和音频流
            import json
            data = json.loads(result.stdout)
            video_streams = [s for s in data['streams'] if s['codec_type'] == 'video']
            audio_streams = [s for s in data['streams'] if s['codec_type'] == 'audio']
            print(f"🎥 视频流数量: {len(video_streams)}")
            print(f"🔊 音频流数量: {len(audio_streams)}")
            
            if video_streams:
                vs = video_streams[0]
                print(f"🎥 视频编码: {vs.get('codec_name', 'unknown')}")
                print(f"🎥 像素格式: {vs.get('pix_fmt', 'unknown')}")
            
            if audio_streams:
                aus = audio_streams[0]
                print(f"🔊 音频编码: {aus.get('codec_name', 'unknown')}")
        else:
            print("❌ FFprobe无法读取文件")
            print(f"错误: {result.stderr}")
    except FileNotFoundError:
        print("⚠️  FFprobe不可用")
    
    return True

def test_video_codecs():
    """测试可用的视频编码器"""
    print("\n=== 测试视频编码器 ===")
    
    codecs_to_test = ['avc1', 'mp4v', 'MJPG', 'XVID', 'H264']
    
    for codec in codecs_to_test:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            # 创建一个测试视频写入器
            test_path = f"test_{codec}.mp4"
            out = cv2.VideoWriter(test_path, fourcc, 30.0, (640, 480))
            
            if out.isOpened():
                print(f"✅ {codec}: 可用")
                # 写入一个测试帧
                import numpy as np
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                success = out.write(test_frame)
                if success:
                    print(f"   ✅ 可以写入帧")
                else:
                    print(f"   ❌ 无法写入帧")
                out.release()
                
                # 清理测试文件
                try:
                    os.remove(test_path)
                except:
                    pass
            else:
                print(f"❌ {codec}: 不可用")
                out.release()
        except Exception as e:
            print(f"❌ {codec}: 错误 - {e}")

def test_ffmpeg():
    """测试FFmpeg可用性"""
    print("\n=== 测试FFmpeg ===")
    
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg可用")
            # 提取版本信息
            version_line = result.stdout.split('\n')[0]
            print(f"📦 版本: {version_line}")
        else:
            print("❌ FFmpeg不可用")
    except FileNotFoundError:
        print("❌ FFmpeg未安装")
    
    try:
        result = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFprobe可用")
        else:
            print("❌ FFprobe不可用")
    except FileNotFoundError:
        print("❌ FFprobe未安装")

def main():
    print("🔍 视频处理调试工具")
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        check_video_info(video_path)
    else:
        print("用法: python debug_video.py <video_file>")
        print("或者直接运行进行系统检查")
    
    # 测试系统组件
    test_video_codecs()
    test_ffmpeg()
    
    print("\n🏁 调试完成")

if __name__ == "__main__":
    main()
